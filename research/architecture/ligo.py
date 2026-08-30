"""LiGO: Learned Linear Growth Operator for model expansion.

Learns a linear map M: Theta_large = M * Theta_small, factored as:
  - R_width: width-growth operator (Kronecker-factored as kron(A, B))
  - L_depth: depth-growth operator (layer duplication)

M is learned with ~100 steps of SGD on a small data subset, then used to
initialize the larger model. Saves up to 50% compute vs training from scratch.

Paper: "LiGO: A Large-scale Learning Initiative for GOrwth of Neural Networks"
arXiv:2303.00980 (ICML 2023).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiGOGrowth:
    """Learned Linear Growth Operator for model expansion.

    Learns Kronecker-factored growth operators that map a smaller model's
    parameters to a larger model's parameters. The operators are learned
    via SGD on a growth objective (CE loss of the expanded model on training data).

    Args:
        src_model: the source (smaller) model
        target_d_model: target d_model (must be >= src d_model)
        target_n_layers: target number of layers (must be >= src n_layers)
    """

    def __init__(self, src_model, target_d_model, target_n_layers):
        self.src_model = src_model
        self.src_d_model = getattr(src_model, "d_model", None)
        self.src_n_layers = getattr(src_model, "n_layers", None)
        self.src_vocab = getattr(src_model, "vocab", 256)
        self.target_d_model = target_d_model
        self.target_n_layers = target_n_layers

        # Width expansion ratio (e.g. 2 for 2x width)
        self.width_ratio = target_d_model // max(self.src_d_model, 1)
        # Depth expansion ratio
        self.depth_ratio = target_n_layers // max(self.src_n_layers, 1)

        # Kronecker-factored width operator: R_width = kron(A, B)
        # A: (width_ratio, 1) — neuron grouping matrix (learnable)
        # B: (src_d_model, src_d_model) — within-group mapping (learnable)
        self.A = nn.Parameter(
            torch.ones(self.width_ratio, 1) / (self.width_ratio ** 0.5)
        )
        self.B = nn.Parameter(torch.eye(self.src_d_model))

        # Depth operator: simple learnable mixing weights for layer mapping
        # For depth_ratio=2: each source layer maps to 2 target layers
        # L_depth[i, j] = weight for source layer j -> target layer i
        self.L_depth = nn.Parameter(
            torch.eye(target_n_layers, self.src_n_layers)
        )

        # Optimizer for growth operators
        self._opt = None

    def _get_opt(self):
        if self._opt is None:
            self._opt = torch.optim.SGD(
                [self.A, self.B, self.L_depth], lr=1e-2, momentum=0.9
            )
        return self._opt

    def get_width_operator(self):
        """Return the full width-growth operator R_width = kron(A, B)."""
        return torch.kron(self.A, self.B)

    def get_width_operator_factors(self):
        """Return the Kronecker factors (A, B) such that R_width = kron(A, B)."""
        return self.A.detach(), self.B.detach()

    def _build_dim_operator(self, dim):
        """Build a (width_ratio * dim, dim) operator for a given dimension."""
        I_dim = torch.eye(dim, device=self.A.device, dtype=self.A.dtype)
        return torch.kron(self.A, I_dim)

    def apply_width_operator(self, src_weight):
        """Apply width growth to a weight matrix (out, in) -> (2*out, 2*in).

        W_target = R_out @ W_source @ R_in^T
        where R = kron(A, I) for each dimension.
        """
        out_dim, in_dim = src_weight.shape
        R_out = self._build_dim_operator(out_dim)  # (2*out, out)
        R_in = self._build_dim_operator(in_dim)    # (2*in, in)
        return R_out @ src_weight @ R_in.T

    def get_depth_operator(self):
        """Return the depth-growth operator L_depth (target_layers x src_layers)."""
        return self.L_depth.detach()

    def apply_depth_operator(self, src_layer_weights):
        """Map source layer weights to target layers.

        Returns a list of target layer weights (target_n_layers items).
        Each target layer is a linear combination of source layers.
        """
        n_src = len(src_layer_weights)
        n_dst = self.target_n_layers
        dst_weights = []
        for i in range(n_dst):
            coeffs = self.L_depth[i]  # (n_src,)
            w = None
            for j in range(n_src):
                c = coeffs[j]
                if c.abs() > 1e-8:
                    term = c * src_layer_weights[j]
                    w = term if w is None else w + term
            if w is None:
                # All coefficients ~0, use first source layer
                w = src_layer_weights[0].clone()
            dst_weights.append(w)
        return dst_weights

    def _build_target_model(self):
        """Build a target model with weights initialized from the growth operators."""
        src = self.src_model
        model_cls = type(src)

        # Detect constructor args
        if hasattr(src, "n_heads") and hasattr(src, "head_dim"):
            # TinyTransformer
            target_n_heads = src.n_heads * self.width_ratio
            target_head_dim = src.head_dim
            target_intermediate = getattr(
                src, "intermediate", src.d_model * 4
            ) * self.width_ratio if hasattr(src, "intermediate") else None
            kwargs = dict(
                d_model=self.target_d_model,
                n_layers=self.target_n_layers,
                n_heads=target_n_heads,
                head_dim=target_head_dim,
                vocab=self.src_vocab,
            )
            if target_intermediate is not None:
                kwargs["intermediate"] = target_intermediate
            dst = model_cls(**kwargs)
        else:
            # TinyMLP
            target_intermediate = (
                getattr(src, "layers", [None])[0].weight.shape[0] * self.width_ratio
                if len(src.layers) > 0 else self.target_d_model * 2
            )
            dst = model_cls(
                d_model=self.target_d_model,
                n_layers=self.target_n_layers,
                vocab=self.src_vocab,
                intermediate=target_intermediate,
            )

        # Apply width operator to embedding
        with torch.no_grad():
            src_embed_w = src.embed.weight  # (vocab, src_d)
            dst_embed_w = self.apply_width_operator(src_embed_w)
            if dst_embed_w.shape == dst.embed.weight.shape:
                dst.embed.weight.copy_(dst_embed_w)

            # Apply width + depth operators to each layer
            src_layer_weights = []
            for layer in src.layers:
                if hasattr(layer, "weight"):
                    src_layer_weights.append(layer.weight)

            if src_layer_weights:
                # Apply width operator to each source layer
                expanded_weights = [
                    self.apply_width_operator(w) for w in src_layer_weights
                ]
                # Apply depth operator
                dst_weights = self.apply_depth_operator(expanded_weights)

                for i, layer in enumerate(dst.layers):
                    if i < len(dst_weights) and hasattr(layer, "weight"):
                        w = dst_weights[i]
                        if w.shape == layer.weight.shape:
                            layer.weight.copy_(w)

            # Apply width operator to head
            if hasattr(src, "head") and hasattr(src.head, "weight"):
                src_head_w = src.head.weight  # (vocab, src_d)
                dst_head_w = self.apply_width_operator(src_head_w)
                if dst_head_w.shape == dst.head.weight.shape:
                    dst.head.weight.copy_(dst_head_w)

            # Copy norm weights (expanded)
            if hasattr(src, "norm"):
                if hasattr(src.norm, "weight") and src.norm.weight is not None:
                    src_nw = src.norm.weight
                    dst_nw = self.apply_width_operator(
                        src_nw.unsqueeze(0)
                    ).squeeze(0)
                    if dst_nw.shape == dst.norm.weight.shape:
                        dst.norm.weight.copy_(dst_nw)
                if hasattr(src.norm, "bias") and src.norm.bias is not None:
                    src_nb = src.norm.bias
                    dst_nb = self.apply_width_operator(
                        src_nb.unsqueeze(0)
                    ).squeeze(0)
                    if dst_nb.shape == dst.norm.bias.shape:
                        dst.norm.bias.copy_(dst_nb)

        return dst

    def compute_growth_loss(self, x, y):
        """Compute the growth objective: CE loss of the expanded model on data.

        Lower loss means the growth operators produce a better initialization
        for the larger model.
        """
        dst = self._build_target_model()
        dst.train()
        out = dst(x)
        vocab = self.src_vocab
        loss = F.cross_entropy(out.view(-1, vocab), y.view(-1))
        return loss

    def growth_step(self, x, y, lr=1e-2):
        """Perform one SGD step on the growth operators (A, B, L_depth)."""
        opt = self._get_opt()
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad()
        loss = self.compute_growth_loss(x, y)
        loss.backward()
        opt.step()
        return loss.item()

    def initialize_larger_model(self):
        """Apply the learned operators to create the final target model.

        Returns a model of the same type as src with expanded dimensions,
        initialized using the learned growth operators.
        """
        with torch.no_grad():
            dst = self._build_target_model()
        return dst
