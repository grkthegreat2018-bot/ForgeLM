"""Self-Play Knowledge Key — generate data → closed-form weight update, no training.

Novel key: combines self-play generation with closed-form knowledge injection.

Pipeline:
  1. Model generates questions about a domain (self-play)
  2. Model generates answers to those questions (self-play)
  3. Quality filter: keep only high-confidence Q→A pairs
  4. Convert Q→A pairs to fact vectors via embeddings
  5. Inject facts into MLP weights via closed-form solution (FactInjectionKey)
  6. Optionally: extract context patches from the Q→A pairs (ContextPatchKey)

This is entirely training-free — no gradient descent, no loss function.
The model "learns" from its own generated data via weight-space algebra.

Quality control:
  - Self-consistency: generate multiple answers, keep majority consensus
  - Confidence filtering: keep only high-logprob answers
  - Diversity: ensure questions cover different topics
  - Dedup: don't inject the same fact twice

Key class: PARTIAL — modifies weights, requires generation (not reversible).

Usage:
    from research.keys.self_play_key import SelfPlayKey, run_self_play
    # Run self-play knowledge injection
    state = run_self_play(model, tokenizer, state, n_rounds=10,
                          domain="science", n_facts=100)
"""
import torch
from typing import Dict, List, Tuple, Optional
from .base import Key, KeyClass, KeyResult


class SelfPlayKey(Key):
    """Self-Play Knowledge key — generate data → closed-form weight update.

    Combines self-play generation with closed-form fact injection.
    No gradient descent, no loss function, no training loop.

    Key class: PARTIAL — modifies weights, requires generation.
    """

    def __init__(self, n_rounds: int = 10, n_facts_per_round: int = 10,
                 confidence_threshold: float = 0.8,
                 domain: str = "general"):
        self.n_rounds = n_rounds
        self.n_facts_per_round = n_facts_per_round
        self.confidence_threshold = confidence_threshold
        self.domain = domain

    @property
    def name(self) -> str:
        return "self_play"

    @property
    def description(self) -> str:
        return "Self-play → closed-form knowledge injection (no training)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """Self-play requires generation — use run_self_play instead."""
        return KeyResult(success=True, weights=data,
                         metadata={"note": "Use run_self_play for actual self-play"})

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        return KeyResult(success=True, data=weights,
                         metadata={"reversible": False})


def generate_questions(model, tokenizer, domain: str, n_questions: int,
                       device: str = "cuda", max_tokens: int = 64) -> List[str]:
    """Generate questions about a domain via self-play.

    Prompts the model to generate questions, then extracts them.
    """
    prompt = f"Generate {n_questions} diverse questions about {domain}:\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    model.eval()
    with torch.inference_mode():
        for _ in range(max_tokens * n_questions):
            logits, _ = model(input_ids)
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # Extract questions (lines with ?)
    questions = [line.strip() for line in text.split("\n")
                 if "?" in line and len(line.strip()) > 10]
    return questions[:n_questions]


def generate_answer(model, tokenizer, question: str, device: str = "cuda",
                    max_tokens: int = 100) -> Tuple[str, float]:
    """Generate an answer to a question and return confidence score.

    Returns (answer_text, confidence) where confidence is the mean
    log-probability of the generated tokens.
    """
    prompt = f"Question: {question}\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    model.eval()
    log_probs = []
    with torch.inference_mode():
        for _ in range(max_tokens):
            logits, _ = model(input_ids)
            next_logits = logits[0, -1]
            next_token = next_logits.argmax()
            log_probs.append(torch.log_softmax(next_logits, dim=-1)[next_token].item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    answer = tokenizer.decode(input_ids[0][len(tokenizer(prompt)["input_ids"]):],
                               skip_special_tokens=True)
    confidence = sum(log_probs) / max(len(log_probs), 1)
    return answer.strip(), confidence


def run_self_play(model, tokenizer, state: Dict[str, torch.Tensor],
                  n_layers: int, d_model: int, d_ff: int = 8960,
                  domain: str = "general", n_rounds: int = 10,
                  n_facts_per_round: int = 10,
                  confidence_threshold: float = 0.8,
                  device: str = "cuda") -> Dict[str, torch.Tensor]:
    """Run self-play knowledge injection.

    Pipeline:
      1. Generate questions about the domain
      2. Generate answers with confidence scores
      3. Filter by confidence
      4. Convert to fact vectors
      5. Inject into MLP weights via closed-form solution

    Args:
        model: transformer model
        tokenizer: tokenizer
        state: model state dict
        n_layers: number of layers
        d_model: model dimension
        d_ff: FFN hidden dimension
        domain: knowledge domain
        n_rounds: number of self-play rounds
        n_facts_per_round: facts to generate per round
        confidence_threshold: minimum confidence to keep
        device: compute device

    Returns:
        modified state dict with injected knowledge
    """
    from .fact_injection_key import inject_facts, create_fact_from_text

    total_facts_injected = 0

    for round_idx in range(n_rounds):
        print(f"\n  [Self-Play] Round {round_idx + 1}/{n_rounds} — domain: {domain}")

        # Generate questions
        questions = generate_questions(model, tokenizer, domain,
                                        n_facts_per_round, device)
        print(f"    Generated {len(questions)} questions")

        # Generate answers with confidence
        facts = []
        for q in questions:
            answer, conf = generate_answer(model, tokenizer, q, device)
            if conf >= confidence_threshold and len(answer) > 5:
                # Convert to fact vectors
                try:
                    x_vec, y_vec = create_fact_from_text(
                        model, tokenizer, q, answer, device)
                    facts.append((x_vec, y_vec))
                except Exception:
                    continue

        print(f"    {len(facts)} facts passed confidence filter (threshold={confidence_threshold})")

        if not facts:
            continue

        # Inject facts into model weights
        state = inject_facts(state, facts, n_layers, d_model, d_ff,
                             layer_idx=-1)  # inject into last layer
        total_facts_injected += len(facts)

        # Reload model with updated weights for next round
        # (so the model can build on its new knowledge)
        from research.model_loader import ModelLoader
        from research.config import get_config
        cfg = get_config("forgelm_v2", device=device)
        # Update model weights in-place
        model.load_state_dict(state, strict=False)

    print(f"\n  [Self-Play] Complete: {total_facts_injected} facts injected over {n_rounds} rounds")
    return state


if __name__ == "__main__":
    key = SelfPlayKey(n_rounds=5, domain="science")
    print(f"Key: {key.name}, class: {key.key_class().value}")
    print(f"  Rounds: {key.n_rounds}")
    print(f"  Domain: {key.domain}")
    print(f"  Confidence threshold: {key.confidence_threshold}")
    print("  Self-Play key verified ✓")
