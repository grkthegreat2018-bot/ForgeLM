"""R39-5: Constrained decoding with XGrammar-style token masking.

XGrammarConstrainer compiles a JSON schema (or a simplified CFG) into a
character-level finite state machine (FSM).  At each decoding step the
FSM state determines which characters are legal, and every token in the
vocabulary whose **first character** is in that allowed set gets a
``True`` in the boolean mask.  Tokens whose first character is not
allowed are masked out (``False``), forcing the model to produce only
schema-conformant JSON.

This is a *simplified* version of full XGrammar — it does not compile a
CFG into a token bitmap.  Instead it uses character-level masking over
a JSON FSM, which is sufficient for tool-call JSON generation where the
schema is a JSON object with typed fields.

States (character-level FSM)::

    START          – expecting the first JSON value
    OBJ_KEY        – after ``{`` or ``,`` in an object: ``"`` or ``}``
    IN_KEY         – inside a key string
    AFTER_KEY      – key string closed: expecting ``:``
    AFTER_COLON    – after ``:``: expecting a value
    IN_STRING      – inside a string value
    IN_NUMBER      – inside a number literal
    IN_TRUE        – partial ``true``
    IN_FALSE       – partial ``false``
    IN_NULL        – partial ``null``
    ARR_VAL        – after ``[`` or ``,`` in an array: value or ``]``
    AFTER_VAL      – value complete: ``,`` / ``}`` / ``]``
    END            – top-level value complete

The context stack (``self._stack``) tracks whether we are inside an
object (``{``) or array (``[``) so that ``AFTER_VAL`` knows which
closing bracket is legal.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

# ── FSM state constants ────────────────────────────────────────────────
START = 0
OBJ_KEY = 1
IN_KEY = 2
AFTER_KEY = 3
AFTER_COLON = 4
IN_STRING = 5
IN_NUMBER = 6
IN_TRUE = 7
IN_FALSE = 8
IN_NULL = 9
ARR_VAL = 10
AFTER_VAL = 11
END = 12

_WHITESPACE = frozenset(" \t\n\r")
_DIGITS = frozenset("0123456789")
_NUMBER_CHARS = frozenset("0123456789.+-eE")
_VALUE_START_CHARS = frozenset('"-0123456789tfn{[')  # chars that start a JSON value
# After a value, structural chars depend on context stack — handled in code.


class XGrammarConstrainer:
    """Compile JSON schema / CFG into a token mask via a character-level FSM.

    Usage::

        constrainer = XGrammarConstrainer(vocab_size, tokenizer)
        constrainer.compile_json(schema)
        constrainer.reset()

        for step in generation_loop:
            mask = constrainer.get_mask(last_token_id)   # (vocab_size,) bool
            logits = logits.masked_fill(~mask, float('-inf'))
            next_token = logits.argmax()
            constrainer.advance(next_token)

    Args:
        vocab_size: size of the model's vocabulary.
        tokenizer: a tokenizer with ``convert_ids_to_tokens(id)`` or
            ``decode([id])``.  Used to map token IDs to their string
            representation so the first character can be checked.
    """

    def __init__(self, vocab_size: int, tokenizer: Any = None):
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self._schema: Optional[dict] = None
        self._grammar_str: Optional[str] = None
        # Precompute first-character of every token for fast masking.
        self._first_chars: list[str] = self._precompute_first_chars()
        # FSM state
        self._state = START
        self._stack: list[str] = []  # '{' or '['
        # Partial keyword tracking for true/false/null
        self._keyword_pos = 0

    # ── Compilation ────────────────────────────────────────────────────

    def compile_json(self, schema: dict) -> "XGrammarConstrainer":
        """Compile a JSON schema into the FSM.

        For the simplified version, the FSM enforces valid JSON structure.
        If the schema specifies a ``type`` (e.g. ``"object"``), the
        top-level state is constrained to only allow that type's starting
        character.
        """
        self._schema = schema
        self._grammar_str = None
        self.reset()
        # Optionally constrain the top-level type.
        top_type = schema.get("type") if isinstance(schema, dict) else None
        if top_type == "object":
            self._top_level_chars = frozenset("{")
        elif top_type == "array":
            self._top_level_chars = frozenset("[")
        elif top_type == "string":
            self._top_level_chars = frozenset('"')
        elif top_type == "number" or top_type == "integer":
            self._top_level_chars = frozenset("-0123456789")
        elif top_type == "boolean":
            self._top_level_chars = frozenset("tf")
        elif top_type == "null":
            self._top_level_chars = frozenset("n")
        else:
            self._top_level_chars = None  # any valid JSON value
        return self

    def compile_grammar(self, grammar_str: str) -> "XGrammarConstrainer":
        """Compile a CFG / grammar string into the FSM.

        Simplified: if the grammar looks JSON-like (contains ``json``,
        ``object``, ``value``), the JSON FSM is used.  Otherwise a
        permissive FSM that allows characters mentioned in the grammar
        is set up.
        """
        self._grammar_str = grammar_str
        self._schema = None
        self.reset()
        lower = grammar_str.lower()
        if any(kw in lower for kw in ("json", "object", "value", "array")):
            self._top_level_chars = None  # JSON FSM
        else:
            # Extract allowed characters from the grammar (simplified).
            allowed = set(c for c in grammar_str if c.isprintable() and not c.isspace())
            self._top_level_chars = frozenset(allowed) if allowed else None
        return self

    # ── Token masking ──────────────────────────────────────────────────

    def get_mask(self, last_token_id: int) -> torch.Tensor:
        """Return a boolean mask ``(vocab_size,)`` of allowed next tokens.

        The mask is ``True`` for tokens whose first character is in the
        allowed set for the current FSM state.
        """
        allowed = self._allowed_chars()
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for tid in range(self.vocab_size):
            fc = self._first_chars[tid]
            if fc in allowed:
                mask[tid] = True
        return mask

    def advance(self, token_id: int) -> None:
        """Update the FSM state after a token is generated.

        Decodes the token to its string representation and feeds each
        character through the character-level FSM.
        """
        text = self._token_to_str(token_id)
        for char in text:
            self._process_char(char)

    def reset(self) -> None:
        """Reset the FSM to its initial state."""
        self._state = START
        self._stack = []
        self._keyword_pos = 0

    # ── Internal: FSM logic ────────────────────────────────────────────

    def _allowed_chars(self) -> frozenset:
        """Return the set of allowed first characters for the current state."""
        s = self._state

        if s == START:
            if self._top_level_chars is not None:
                return self._top_level_chars | _WHITESPACE
            return _VALUE_START_CHARS | _WHITESPACE

        if s == OBJ_KEY:
            return frozenset('"}') | _WHITESPACE

        if s == IN_KEY:
            # Inside a key string: any char is allowed (including " to close).
            # In a real tokenizer, " inside a string would be escaped as \",
            # but for character-level masking we allow all chars.
            return frozenset(chr(i) for i in range(32, 127))

        if s == AFTER_KEY:
            return frozenset(":") | _WHITESPACE

        if s == AFTER_COLON:
            return _VALUE_START_CHARS | _WHITESPACE

        if s == IN_STRING:
            return frozenset(chr(i) for i in range(32, 127))

        if s == IN_NUMBER:
            # Number continuation + structural chars that end the number.
            struct = self._structural_after_value()
            return _NUMBER_CHARS | struct | _WHITESPACE

        if s == IN_TRUE:
            # "true" — next expected char depends on position
            expected = "true"[self._keyword_pos] if self._keyword_pos < 4 else ""
            if expected:
                return frozenset(expected)
            return self._structural_after_value() | _WHITESPACE

        if s == IN_FALSE:
            expected = "false"[self._keyword_pos] if self._keyword_pos < 5 else ""
            if expected:
                return frozenset(expected)
            return self._structural_after_value() | _WHITESPACE

        if s == IN_NULL:
            expected = "null"[self._keyword_pos] if self._keyword_pos < 4 else ""
            if expected:
                return frozenset(expected)
            return self._structural_after_value() | _WHITESPACE

        if s == ARR_VAL:
            return _VALUE_START_CHARS | frozenset("]") | _WHITESPACE

        if s == AFTER_VAL:
            return self._structural_after_value() | _WHITESPACE

        if s == END:
            return _WHITESPACE  # only whitespace after top-level value

        # Fallback: allow everything (shouldn't happen).
        return frozenset(chr(i) for i in range(32, 127))

    def _structural_after_value(self) -> frozenset:
        """Return the structural characters allowed after a complete value."""
        allowed = set()
        if self._stack:
            top = self._stack[-1]
            if top == "{":
                allowed.add(",")
                allowed.add("}")
            elif top == "[":
                allowed.add(",")
                allowed.add("]")
        else:
            # Top-level: value is complete.
            pass
        return frozenset(allowed)

    def _process_char(self, char: str) -> None:
        """Feed a single character through the FSM, updating state."""
        s = self._state

        # Whitespace is generally ignored (except in IN_STRING/IN_KEY).
        if char in _WHITESPACE and s not in (IN_STRING, IN_KEY):
            return

        if s == START:
            self._process_value_start(char)

        elif s == OBJ_KEY:
            if char == '"':
                self._state = IN_KEY
            elif char == "}":
                self._pop_and_after_value()

        elif s == IN_KEY:
            if char == '"':
                self._state = AFTER_KEY
            # else: stay in IN_KEY (string content)

        elif s == AFTER_KEY:
            if char == ":":
                self._state = AFTER_COLON

        elif s == AFTER_COLON:
            self._process_value_start(char)

        elif s == IN_STRING:
            if char == '"':
                self._state = AFTER_VAL
            # else: stay in IN_STRING

        elif s == IN_NUMBER:
            if char in _NUMBER_CHARS:
                pass  # stay in IN_NUMBER
            elif char == ",":
                self._handle_comma()
            elif char == "}":
                self._pop_and_after_value()
            elif char == "]":
                self._pop_and_after_value()
            # else: stay (shouldn't happen with masking)

        elif s == IN_TRUE:
            self._process_keyword(char, "true", IN_TRUE)

        elif s == IN_FALSE:
            self._process_keyword(char, "false", IN_FALSE)

        elif s == IN_NULL:
            self._process_keyword(char, "null", IN_NULL)

        elif s == ARR_VAL:
            if char == "]":
                self._pop_and_after_value()
            else:
                self._process_value_start(char)

        elif s == AFTER_VAL:
            if char == ",":
                self._handle_comma()
            elif char == "}":
                self._pop_and_after_value()
            elif char == "]":
                self._pop_and_after_value()

        elif s == END:
            pass  # only whitespace

    def _process_value_start(self, char: str) -> None:
        """Process the start of a JSON value from START/AFTER_COLON/ARR_VAL."""
        if char == "{":
            self._stack.append("{")
            self._state = OBJ_KEY
        elif char == "[":
            self._stack.append("[")
            self._state = ARR_VAL
        elif char == '"':
            self._state = IN_STRING
        elif char in _DIGITS or char == "-":
            self._state = IN_NUMBER
            self._keyword_pos = 0
        elif char == "t":
            self._state = IN_TRUE
            self._keyword_pos = 1  # we've seen 't'
        elif char == "f":
            self._state = IN_FALSE
            self._keyword_pos = 1  # we've seen 'f'
        elif char == "n":
            self._state = IN_NULL
            self._keyword_pos = 1  # we've seen 'n'

    def _process_keyword(self, char: str, keyword: str, state: int) -> None:
        """Process a character while matching a keyword (true/false/null)."""
        if self._keyword_pos < len(keyword) and char == keyword[self._keyword_pos]:
            self._keyword_pos += 1
            if self._keyword_pos >= len(keyword):
                self._state = AFTER_VAL
        # If char doesn't match, stay in state (masking should prevent this)

    def _handle_comma(self) -> None:
        """Handle a comma — transition to the next key (object) or value (array)."""
        if self._stack and self._stack[-1] == "{":
            self._state = OBJ_KEY
        elif self._stack and self._stack[-1] == "[":
            self._state = ARR_VAL

    def _pop_and_after_value(self) -> None:
        """Pop the context stack and transition to AFTER_VAL or END."""
        if self._stack:
            self._stack.pop()
        if self._stack:
            self._state = AFTER_VAL
        else:
            self._state = END

    # ── Internal: token utilities ──────────────────────────────────────

    def _precompute_first_chars(self) -> list[str]:
        """Precompute the first character of every token in the vocabulary."""
        first_chars = []
        for tid in range(self.vocab_size):
            first_chars.append(self._token_to_str(tid)[:1] if self.tokenizer else "")
        return first_chars

    def _token_to_str(self, token_id: int) -> str:
        """Decode a single token ID to its string representation."""
        if self.tokenizer is None:
            return ""
        # Try convert_ids_to_tokens (HF tokenizers)
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            try:
                tok = self.tokenizer.convert_ids_to_tokens(token_id)
                if isinstance(tok, list):
                    return "".join(str(t) for t in tok)
                return str(tok)
            except Exception:
                pass
        # Fallback: decode([id])
        if hasattr(self.tokenizer, "decode"):
            try:
                return self.tokenizer.decode([token_id])
            except Exception:
                pass
        return ""
