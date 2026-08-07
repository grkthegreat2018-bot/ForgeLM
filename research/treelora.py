"""TreeLoRA — Hierarchical LoRA adapters for multi-task continual learning.

Organizes LoRA adapters in a tree structure based on gradient similarity:
- Similar tasks share lower-level adapters (tree branches)
- Dissimilar tasks get separate branches
- New tasks can attach to the most similar existing branch

This prevents catastrophic forgetting across many tasks while keeping
adapter count manageable (shared base adapters + task-specific leaves).

Tree structure:
    root (base LoRA)
    ├── coding (shared coding adapter)
    │   ├── python (python-specific)
    │   └── rust (rust-specific)
    ├── math (shared math adapter)
    │   ├── algebra
    │   └── calculus
    └── writing (shared writing adapter)

Usage:
    from research.treelora import TreeLoRAManager
    mgr = TreeLoRAManager(model, d_model=1024, lora_rank=16)
    mgr.add_task("python", parent="coding")
    mgr.train_task("python", data, steps=100)
    mgr.save_tree("research/checkpoints/tree_lora/")
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional

from research.live_learn import LoRALinear, inject_lora, extract_lora_state_dict


class TreeNode:
    """A node in the TreeLoRA hierarchy."""

    def __init__(self, name, parent: Optional[str] = None,
                 lora_state_dict: Optional[dict] = None):
        self.name = name
        self.parent = parent
        self.children: List[str] = []
        self.lora_state_dict = lora_state_dict or {}
        self.gradient_signature: Optional[torch.Tensor] = None
        self.metadata: dict = {}

    def to_dict(self):
        return {
            "name": self.name,
            "parent": self.parent,
            "children": self.children,
            "metadata": self.metadata,
        }


class TreeLoRAManager:
    """Manages a tree of LoRA adapters for multi-task learning.

    Args:
        model: the base model (LoRA is injected)
        d_model: model dimension
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        similarity_threshold: gradient similarity threshold for auto-parenting
    """

    def __init__(self, model, d_model=1024, lora_rank=16, lora_alpha=32,
                 similarity_threshold=0.7):
        self.model = model
        self.d_model = d_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.similarity_threshold = similarity_threshold

        # Inject LoRA into model.
        inject_lora(model, rank=lora_rank, alpha=lora_alpha)

        # Tree structure.
        self.nodes: Dict[str, TreeNode] = {}
        self.root = TreeNode("root")
        self.nodes["root"] = self.root

        # Current active adapters (path from root to active leaf).
        self.active_path: List[str] = ["root"]

    def add_task(self, name: str, parent: Optional[str] = None,
                 auto_parent: bool = True, data_batch=None) -> str:
        """Add a new task to the tree.

        Args:
            name: task name
            parent: explicit parent task name. If None and auto_parent=True,
                    find the most similar existing task.
            auto_parent: if True and no parent given, use gradient similarity
            data_batch: optional (input_ids, targets) for gradient-based auto-parenting

        Returns:
            the parent name that was assigned
        """
        if name in self.nodes:
            return self.nodes[name].parent or "root"

        if parent is None and auto_parent:
            parent = self._find_most_similar(name, data_batch=data_batch)
        elif parent is None:
            parent = "root"

        if parent not in self.nodes:
            raise ValueError(f"Parent '{parent}' not found in tree")

        node = TreeNode(name, parent=parent)
        node.lora_state_dict = extract_lora_state_dict(self.model)
        self.nodes[name] = node
        self.nodes[parent].children.append(name)
        print(f"  [TreeLoRA] added '{name}' under '{parent}'")
        return parent

    def _find_most_similar(self, name: str, data_batch=None) -> str:
        """Find the most similar existing task for auto-parenting.

        Two-stage approach:
        1. If data_batch is provided, compute gradient signature and compare
           cosine similarity with existing nodes' signatures.
        2. Fall back to name-based keyword matching if no data or no signatures.

        Args:
            name: new task name
            data_batch: optional (input_ids, targets) for gradient signature

        Returns:
            name of the most similar existing node (parent)
        """
        # Stage 1: gradient-based similarity (if data and existing signatures).
        if data_batch is not None:
            new_sig = self.compute_gradient_signature(data_batch)
            if new_sig.numel() > 0:
                best_sim = -1.0
                best_node = "root"
                for node_name, node in self.nodes.items():
                    if node.gradient_signature is None:
                        continue
                    # Cosine similarity.
                    sig = node.gradient_signature
                    if sig.numel() != new_sig.numel():
                        continue
                    sim = torch.nn.functional.cosine_similarity(
                        new_sig.unsqueeze(0), sig.unsqueeze(0)
                    ).item()
                    if sim > best_sim and sim >= self.similarity_threshold:
                        best_sim = sim
                        best_node = node_name
                if best_sim >= self.similarity_threshold:
                    print(f"  [TreeLoRA] gradient similarity: {name} → {best_node} (cos={best_sim:.3f})")
                    return best_node

        # Stage 2: name-based keyword matching fallback.
        categories = {
            "coding": ["python", "rust", "java", "c++", "javascript", "go", "code"],
            "math": ["algebra", "calculus", "geometry", "statistics", "math"],
            "writing": ["essay", "story", "creative", "writing", "blog"],
            "reasoning": ["logic", "puzzle", "reasoning", "analysis"],
        }

        name_lower = name.lower()
        for category, keywords in categories.items():
            if any(kw in name_lower for kw in keywords):
                if category in self.nodes:
                    return category

        return "root"

    def compute_gradient_signature(self, data_batch) -> torch.Tensor:
        """Compute a gradient signature for a task.

        Flattens all LoRA gradients into a single vector — used for
        similarity comparison between tasks.
        """
        self.model.train()
        self.model.zero_grad()

        # Forward + backward on a small batch.
        input_ids, targets = data_batch
        out = self.model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )
        loss.backward()

        # Collect LoRA gradients.
        grads = []
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.flatten())

        signature = torch.cat(grads) if grads else torch.tensor([])
        self.model.zero_grad()
        return signature

    def activate_path(self, task_name: str):
        """Activate the adapter path from root to the given task.

        This blends all adapters along the path (root → ... → task).
        """
        if task_name not in self.nodes:
            raise ValueError(f"Task '{task_name}' not found")

        # Build path from root to task.
        path = []
        current = task_name
        while current is not None:
            path.append(current)
            current = self.nodes[current].parent
        path.reverse()  # root → ... → task
        self.active_path = path

        # Blend adapters along the path (weighted sum, root gets least weight).
        n = len(path)
        weights = [1.0 / (n - i) for i in range(n)]  # root=1/n, leaf=1
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # Weighted sum of LoRA state dicts.
        blended = {}
        for i, node_name in enumerate(path):
            sd = self.nodes[node_name].lora_state_dict
            w = weights[i]
            for key in sd:
                if key not in blended:
                    blended[key] = w * sd[key]
                else:
                    blended[key] = blended[key] + w * sd[key]

        # Apply blended adapters to model.
        for name, param in self.model.named_parameters():
            if name in blended:
                param.data.copy_(blended[name])

        print(f"  [TreeLoRA] activated path: {' → '.join(path)}")

    def train_task(self, task_name: str, data_loader, steps=100, lr=1e-4):
        """Train the LoRA adapter for a specific task.

        Only the leaf adapter is trained; parent adapters are frozen.
        """
        if task_name not in self.nodes:
            raise ValueError(f"Task '{task_name}' not found. Add it first.")

        self.activate_path(task_name)

        # Freeze all params except LoRA (already done by inject_lora).
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params, lr=lr)

        self.model.train()
        step = 0
        while step < steps:
            for batch in data_loader:
                if step >= steps:
                    break
                input_ids, targets = batch
                out = self.model(input_ids)
                logits = out[0] if isinstance(out, tuple) else out
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-100,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                optimizer.step()
                step += 1
                if step % 20 == 0:
                    print(f"  [TreeLoRA] {task_name} step {step}/{steps} | loss: {loss.item():.4f}")

        # Save the trained adapter.
        self.nodes[task_name].lora_state_dict = extract_lora_state_dict(self.model)
        print(f"  [TreeLoRA] saved adapter for '{task_name}'")

    def save_tree(self, dirpath):
        """Save the entire tree structure + all adapters."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        # Save tree structure.
        tree_dict = {name: node.to_dict() for name, node in self.nodes.items()}
        with open(dirpath / "tree.json", "w") as f:
            json.dump(tree_dict, f, indent=2)

        # Save each adapter.
        for name, node in self.nodes.items():
            if node.lora_state_dict:
                adapter_dir = dirpath / name
                adapter_dir.mkdir(exist_ok=True)
                try:
                    from safetensors.torch import save_file
                    save_file(node.lora_state_dict, str(adapter_dir / "adapter.safetensors"))
                except ImportError:
                    torch.save(node.lora_state_dict, adapter_dir / "adapter.pt")

        print(f"  [TreeLoRA] saved tree to {dirpath}")

    def load_tree(self, dirpath):
        """Load a tree from disk."""
        dirpath = Path(dirpath)
        with open(dirpath / "tree.json") as f:
            tree_dict = json.load(f)

        self.nodes = {}
        for name, data in tree_dict.items():
            node = TreeNode(name, parent=data["parent"])
            node.children = data["children"]
            node.metadata = data.get("metadata", {})

            # Load adapter if exists.
            adapter_dir = dirpath / name
            if adapter_dir.exists():
                adapter_file = adapter_dir / "adapter.safetensors"
                if adapter_file.exists():
                    from safetensors.torch import load_file
                    node.lora_state_dict = load_file(str(adapter_file))
                else:
                    adapter_file = adapter_dir / "adapter.pt"
                    if adapter_file.exists():
                        node.lora_state_dict = torch.load(adapter_file, map_location="cpu")

            self.nodes[name] = node

        self.root = self.nodes.get("root", TreeNode("root"))
        print(f"  [TreeLoRA] loaded tree with {len(self.nodes)} nodes from {dirpath}")

    def tree_summary(self):
        """Print a visual tree summary."""
        def _print_node(name, indent=0):
            node = self.nodes[name]
            prefix = "  " * indent + ("├── " if indent > 0 else "")
            n_params = sum(t.numel() for t in node.lora_state_dict.values())
            print(f"{prefix}{name} ({n_params:,} params, {len(node.children)} children)")
            for child in node.children:
                _print_node(child, indent + 1)

        _print_node("root")
