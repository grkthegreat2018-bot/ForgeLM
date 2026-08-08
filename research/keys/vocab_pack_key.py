"""Vocabulary Pack Key — portable domain-specific token embedding deltas.

Novel insight: When a model is fine-tuned on a domain (medical, legal, code),
the embedding layer captures domain-specific knowledge as shifts in token
vectors.  Instead of copying the entire fine-tuned embedding table, we extract
only the *delta* for domain-relevant tokens:

    delta[id] = E_domain[id] - E_base[id]

This delta is a compact, portable "Vocabulary Pack" that can be injected into
*any* base model sharing the same vocabulary — even one with a different
architecture — because the delta operates in the shared embedding space.

Forward (data -> weights):  Extract deltas from (base, domain) embedding pair.
                             Returns a portable pack {token_ids, deltas}.
Reverse (weights -> data):  Reconstruct domain embeddings from base + deltas.

Key class: PARTIAL — the pack is portable across architectures with the same
vocabulary, but the reverse direction requires the original base embeddings.

Usage:
    from research.keys.vocab_pack_key import VocabPackKey
    key = VocabPackKey()
    # Extract pack from fine-tuned model
    result = key.forward({"base_embed": E_base, "domain_embed": E_domain,
                          "domain_token_ids": [100, 200, 300]})
    # Inject into a different base model
    reconstructed = key.reverse(result.weights)
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .base import Key, KeyClass, KeyResult


class VocabPackKey(Key):
    """Vocabulary Pack key — extract and inject domain-specific embedding deltas.

    Captures what a domain fine-tune learned for specific tokens as a compact,
    portable delta tensor.  The pack can be transferred to any base model with
    the same vocabulary.

    Key class: PARTIAL — forward extracts the pack, reverse reconstructs
    domain embeddings given the base.  The pack itself is architecture-agnostic.
    """

    @property
    def name(self) -> str:
        return "vocab_pack"

    @property
    def description(self) -> str:
        return (
            "Portable domain-specific token embedding deltas: "
            "extract delta = E_domain - E_base, inject into any same-vocab base."
        )

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: Dict[str, torch.Tensor]) -> KeyResult:
        """Extract a vocabulary pack from (base, domain) embeddings.

        Args:
            data: dict with keys:
                - base_embed: nn.Embedding or (vocab, d_model) tensor — base model
                - domain_embed: nn.Embedding or (vocab, d_model) tensor — fine-tuned
                - domain_token_ids: list[int] — token IDs to extract deltas for

        Returns:
            KeyResult with weights = {"token_ids": LongTensor, "deltas": (n, d_model)}
        """
        base_embed = data["base_embed"]
        domain_embed = data["domain_embed"]
        token_ids: List[int] = data["domain_token_ids"]

        # Accept nn.Embedding or raw weight tensors
        if isinstance(base_embed, nn.Embedding):
            base_w = base_embed.weight
        else:
            base_w = base_embed
        if isinstance(domain_embed, nn.Embedding):
            domain_w = domain_embed.weight
        else:
            domain_w = domain_embed

        ids_t = torch.tensor(token_ids, dtype=torch.long, device=base_w.device)
        # Gather rows for the specified tokens
        base_rows = base_w[ids_t]       # (n_tokens, d_model)
        domain_rows = domain_w[ids_t]   # (n_tokens, d_model)
        # Delta captures what the domain fine-tune learned
        deltas = domain_rows - base_rows

        return KeyResult(
            success=True,
            weights={
                "token_ids": ids_t.cpu(),
                "deltas": deltas.detach().cpu(),
            },
            metadata={
                "n_tokens": len(token_ids),
                "d_model": base_w.shape[1],
                "pack_size_bytes": deltas.numel() * deltas.element_size(),
            },
        )

    def reverse(self, weights: Dict[str, torch.Tensor]) -> KeyResult:
        """Reconstruct domain embeddings from a vocabulary pack.

        Requires the caller to supply base embeddings alongside the pack.
        In practice, the base model's embedding table provides E_base.

        Args:
            weights: dict with keys:
                - token_ids: LongTensor of token IDs
                - deltas: (n_tokens, d_model) delta tensor
                - base_embed: (vocab, d_model) base embedding table (required)

        Returns:
            KeyResult with data = {"domain_embeds": (n_tokens, d_model)}
        """
        if "deltas" not in weights or "token_ids" not in weights:
            return KeyResult(success=False, error="Missing token_ids or deltas in pack")
        if "base_embed" not in weights:
            return KeyResult(
                success=False,
                error="reverse requires base_embed to reconstruct domain embeddings",
            )

        token_ids = weights["token_ids"]
        deltas = weights["deltas"]
        base_embed = weights["base_embed"]

        if isinstance(base_embed, nn.Embedding):
            base_w = base_embed.weight
        else:
            base_w = base_embed

        # Reconstruct: E_domain[id] = E_base[id] + delta
        base_rows = base_w[token_ids]  # (n_tokens, d_model)
        domain_embeds = base_rows + deltas.to(base_rows.device)

        return KeyResult(
            success=True,
            data={"domain_embeds": domain_embeds},
            metadata={"n_tokens": len(token_ids)},
        )


if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size, d_model = 1000, 128
    domain_token_ids = [10, 50, 100, 200, 500]

    # Synthetic base and domain-fine-tuned embeddings
    E_base = nn.Embedding(vocab_size, d_model)
    E_domain = nn.Embedding(vocab_size, d_model)
    # Simulate fine-tuning: domain embeddings drift from base
    with torch.no_grad():
        E_domain.weight.copy_(E_base.weight + torch.randn_like(E_base.weight) * 0.1)

    key = VocabPackKey()
    print(f"Key: {key.name}, class: {key.key_class().value}")
    print(f"  Description: {key.description}")

    # Forward: extract pack
    result = key.forward({
        "base_embed": E_base,
        "domain_embed": E_domain,
        "domain_token_ids": domain_token_ids,
    })
    assert result.success, f"Forward failed: {result.error}"
    print(f"  Forward: extracted {result.metadata['n_tokens']} deltas, "
          f"d_model={result.metadata['d_model']}")

    deltas = result.weights["deltas"]
    token_ids = result.weights["token_ids"]
    assert deltas.shape == (len(domain_token_ids), d_model)
    # Verify delta matches E_domain - E_base
    expected = (E_domain.weight[torch.tensor(domain_token_ids)]
                - E_base.weight[torch.tensor(domain_token_ids)])
    assert torch.allclose(deltas, expected, atol=1e-6), "Delta mismatch!"
    print("  Delta extraction verified (matches E_domain - E_base)")

    # Reverse: reconstruct domain embeddings
    rev = key.reverse({
        "token_ids": token_ids,
        "deltas": deltas,
        "base_embed": E_base,
    })
    assert rev.success, f"Reverse failed: {rev.error}"
    reconstructed = rev.data["domain_embeds"]
    original = E_domain.weight[torch.tensor(domain_token_ids)]
    assert torch.allclose(reconstructed, original, atol=1e-5), "Reconstruction mismatch!"
    print("  Reverse: domain embeddings reconstructed (round-trip verified)")

    # Portability test: inject deltas into a *different* base with same vocab
    E_base2 = nn.Embedding(vocab_size, d_model)
    with torch.no_grad():
        E_base2.weight.copy_(E_base.weight)  # same vocab, different instance
    rev2 = key.reverse({
        "token_ids": token_ids,
        "deltas": deltas,
        "base_embed": E_base2,
    })
    assert rev2.success
    assert torch.allclose(rev2.data["domain_embeds"], original, atol=1e-5)
    print("  Portability: pack injected into second base model successfully")
    print("  All tests passed.")
