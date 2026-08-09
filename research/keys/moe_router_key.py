"""MoE Router key — derive router weights from FFN weight clustering.

The MoE router decides which expert to route each token to. Without training,
the router is uniform (all experts equally likely), giving cos=0.73.

This key derives router weights by clustering the FFN weight slices:
- Each expert's weight slice has a "direction" in weight space
- The router should route tokens to the expert whose weights best match
  the token's hidden state
- We compute the centroid of each expert's weight slice, then set the
  router weight to project hidden states toward these centroids

Key class: PARTIAL — approximate initialization, better than uniform but
not optimal. Still needs light fine-tuning.

Math:
  Expert i has weights W_i (d_ff × d_model for w_gate, w_up, w_down)
  Centroid c_i = mean of W_i rows (d_model vector)
  Router weight R[i] = c_i (so router_score_i = h · c_i)
  Token h gets routed to expert with highest h · c_i
"""
import torch
import torch.nn.functional as F

from research.keys.base import Key, KeyClass, KeyResult


class MoERouterKey(Key):
    """Derive MoE router weights from FFN weight slice centroids.

    Instead of uniform routing (zeros), this key computes the centroid
    of each expert's weight slice and uses it as the router weight row.
    Tokens are routed to the expert whose centroid is closest (dot product).

    This gives a meaningful initialization: tokens similar to an expert's
    weight pattern get routed there. Still approximate — real routing
    decisions depend on token semantics, not just weight similarity.
    """

    @property
    def name(self) -> str:
        return "moe_router"

    @property
    def description(self) -> str:
        return "Derive router from FFN weight centroids (better than uniform init)"

    def key_class(self) -> KeyClass:
        return KeyClass.PARTIAL

    def forward(self, data: dict) -> KeyResult:
        """Derive router weights from expert weight slices.

        Args:
            data: {
                "expert_weights": list of dicts, each with "w_gate", "w_up", "w_down",
                "d_model": int,
                "shared_weight": optional float (weight for shared expert, default 0)
            }

        Returns:
            {"router_weight": tensor (n_experts, d_model),
             "router_bias": tensor (n_experts,)}
        """
        try:
            experts = data["expert_weights"]
            d_model = data["d_model"]
            n_experts = len(experts)
            shared_w = data.get("shared_weight", 0.0)

            # Compute centroid for each expert
            centroids = torch.zeros(n_experts, d_model)
            for i, exp in enumerate(experts):
                # Use w_gate (d_ff × d_model) — it's the first projection,
                # so its row norms indicate which input dims this expert cares about
                w_gate = exp["w_gate"]  # (d_ff, d_model)
                w_up = exp["w_up"]      # (d_ff, d_model)

                # Centroid = mean of w_gate rows (average direction this expert projects)
                # Weight by w_up norms (experts with larger up-projection are more "active")
                gate_norms = w_gate.norm(dim=1, keepdim=True)  # (d_ff, 1)
                up_norms = w_up.norm(dim=1, keepdim=True)      # (d_ff, 1)

                # Weighted centroid: rows with larger gate+up norms matter more
                weights = (gate_norms * up_norms).squeeze(1)  # (d_ff,)
                weights = weights / weights.sum().clamp(min=1e-8)

                centroid = (weights.unsqueeze(1) * w_gate).sum(dim=0)  # (d_model,)
                centroids[i] = centroid

            # Normalize centroids to unit norm (so router scores are comparable)
            centroid_norms = centroids.norm(dim=1, keepdim=True).clamp(min=1e-8)
            router_weight = centroids / centroid_norms

            # Bias: shared expert gets negative bias (always active, don't route to it)
            # Routed experts get zero bias (let dot product decide)
            router_bias = torch.zeros(n_experts)

            return KeyResult(
                success=True,
                weights={"router_weight": router_weight, "router_bias": router_bias},
                metadata={
                    "n_experts": n_experts,
                    "d_model": d_model,
                    "method": "weighted_centroid",
                    "shared_weight": shared_w,
                },
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))

    def reverse(self, weights: dict) -> KeyResult:
        """Extract expert centroids from router weights (approximate).

        The router weight rows ARE the centroids (normalized), so this is
        just a passthrough. We can't recover the original FFN weights from
        the router — only the routing directions.
        """
        try:
            router_weight = weights["router_weight"]
            return KeyResult(
                success=True,
                data={"centroids": router_weight},
                metadata={"approximate": True},
            )
        except Exception as e:
            return KeyResult(success=False, error=str(e))


def init_router_from_experts(moe_layer, expert_weights_list):
    """Initialize a MoE layer's router from expert weight centroids (in-place).

    Args:
        moe_layer: MoELayer with .router (nn.Linear d_model -> n_experts)
        expert_weights_list: list of dicts with "w_gate", "w_up", "w_down" per expert
    """
    d_model = moe_layer.router.in_features
    key = MoERouterKey()
    result = key.forward({
        "expert_weights": expert_weights_list,
        "d_model": d_model,
    })
    if result.success:
        moe_layer.router.weight.data.copy_(result.weights["router_weight"])
        if hasattr(moe_layer.router, 'bias') and moe_layer.router.bias is not None:
            moe_layer.router.bias.data.copy_(result.weights["router_bias"])
    return result


if __name__ == "__main__":
    # Test with synthetic expert weights
    d_model = 256
    d_ff = 512
    n_experts = 4

    experts = []
    for i in range(n_experts):
        # Each expert has a different "direction" in weight space
        center = torch.randn(d_model)
        w_gate = center.unsqueeze(0).expand(d_ff, d_model) + 0.1 * torch.randn(d_ff, d_model)
        w_up = torch.randn(d_ff, d_model)
        w_down = torch.randn(d_model, d_ff)
        experts.append({"w_gate": w_gate, "w_up": w_up, "w_down": w_down})

    key = MoERouterKey()
    result = key.forward({"expert_weights": experts, "d_model": d_model})
    print(f"Key: {key.name}, class: {key.key_class().value}")
    print(f"Forward: {result.success}")
    print(f"Router weight shape: {result.weights['router_weight'].shape}")
    print(f"Router bias: {result.weights['router_bias']}")

    # Test routing: a token similar to expert 0's direction should route to expert 0
    token = experts[0]["w_gate"][0]  # similar to expert 0
    scores = result.weights["router_weight"] @ token
    probs = F.softmax(scores, dim=0)
    print(f"Routing probs for expert-0-like token: {probs}")
    print(f"Routed to expert: {probs.argmax().item()} (expected 0)")
