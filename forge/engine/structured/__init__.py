"""Structured output / constrained decoding package.

R39-5: Constrained decoding with XGrammar-style token masking.
  - XGrammarConstrainer: compiles JSON schema / CFG into a token mask
    via a character-level finite state machine.

The constrainer enforces JSON schema conformance at the token level by
tracking an FSM state and masking tokens whose first character is not
in the allowed set for the current state. This is sufficient for
tool-call JSON generation — full XGrammar (CFG → token bitmap) is not
needed for the structured-output use case.
"""
from .xgrammar import XGrammarConstrainer

__all__ = ["XGrammarConstrainer"]
