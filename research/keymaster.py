"""KEYMASTER — the orchestrator for keys and keystacks.

Usage:
  # List available keys
  python -m research.keymaster --list-keys

  # List available keystacks
  python -m research.keymaster --list-keystacks

  # Describe a keystack
  python -m research.keymaster --describe qwen2

  # Forward: data -> weights (using a single key)
  python -m research.keymaster --key linear_mse --forward --data data.pt --out weights.safetensors

  # Forward: data -> weights (using a keystack)
  python -m research.keymaster --keystack qwen2 --forward --data data.pt --out weights.safetensors

  # Reverse: weights -> data
  python -m research.keymaster --keystack qwen2 --reverse --weights model.safetensors --out data.pt

  # Cross-arch: weights(A) -> weights(B)
  python -m research.keymaster --keystack qwen2 --cross-arch --target-keystack deepseek \
      --weights model_a.safetensors --out model_b.safetensors

  # Test: verify a key against trained weights
  python -m research.keymaster --key linear_mse --test
"""
import argparse
import sys
import torch
from typing import Dict, Optional

from research.keys import (
    KEY_REGISTRY, KEYSTACK_REGISTRY, KeyClass, KeyResult,
    build_qwen2_keystack,
)


def list_keys():
    """Print all available keys."""
    print("\nAvailable Keys:")
    print(f"{'Name':<20} {'Class':<10} {'Description'}")
    print("-" * 70)
    for name, cls in sorted(KEY_REGISTRY.items()):
        key = cls()
        print(f"{name:<20} {key.key_class().value:<10} {key.description}")
    print()


def list_keystacks():
    """Print all available keystacks."""
    print("\nAvailable KeyStacks:")
    print(f"{'Name':<20} {'Components'}")
    print("-" * 70)
    for name, builder in sorted(KEYSTACK_REGISTRY.items()):
        stack = builder()
        components = ", ".join(k.name for k in stack.keys)
        print(f"{name:<20} {components}")
    print()


def describe_keystack(name: str):
    """Print detailed description of a keystack."""
    if name not in KEYSTACK_REGISTRY:
        print(f"Error: KeyStack '{name}' not found. Available: {list(KEYSTACK_REGISTRY.keys())}")
        return
    stack = KEYSTACK_REGISTRY[name]()
    print(stack.describe())
    print()


def load_data(path: str) -> Dict[str, torch.Tensor]:
    """Load data from .pt or .safetensors file."""
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(path)
    else:
        return torch.load(path, weights_only=False)


def save_result(result: KeyResult, path: str):
    """Save key result (weights or data) to file."""
    if result.weights is not None:
        if path.endswith('.safetensors'):
            from safetensors.torch import save_file
            save_file(result.weights, path)
        else:
            torch.save(result.weights, path)
        print(f"Saved weights to {path}")
    elif result.data is not None:
        if path.endswith('.safetensors'):
            from safetensors.torch import save_file
            # Filter to tensors only
            tensor_data = {k: v for k, v in result.data.items() if isinstance(v, torch.Tensor)}
            save_file(tensor_data, path)
        else:
            torch.save(result.data, path)
        print(f"Saved data to {path}")
    else:
        print("Nothing to save (no weights or data in result)")


def run_forward(key_or_stack, data: Dict[str, torch.Tensor], out_path: str):
    """Forward: data -> weights."""
    print(f"\nForward: {key_or_stack}")
    print(f"  Input data keys: {list(data.keys())}")

    result = key_or_stack.forward(data)

    if result.success:
        print(f"  SUCCESS: produced {len(result.weights)} weight tensors")
        for name, tensor in result.weights.items():
            if isinstance(tensor, torch.Tensor):
                print(f"    {name}: {list(tensor.shape)}")
        if result.metadata:
            for k, v in result.metadata.items():
                if k != 'errors':
                    print(f"    [{k}]: {v}")
        if result.metadata.get('errors'):
            print(f"  Warnings: {result.metadata['errors']}")
        if out_path:
            save_result(result, out_path)
    else:
        print(f"  FAILED: {result.error}")
        sys.exit(1)

    return result


def run_reverse(key_or_stack, weights: Dict[str, torch.Tensor], out_path: str):
    """Reverse: weights -> data."""
    print(f"\nReverse: {key_or_stack}")
    print(f"  Input weight keys: {list(weights.keys())}")

    result = key_or_stack.reverse(weights)

    if result.success:
        print(f"  SUCCESS: extracted {len(result.data)} data items")
        for name, val in result.data.items():
            if isinstance(val, torch.Tensor):
                print(f"    {name}: {list(val.shape)}")
        if result.metadata:
            for k, v in result.metadata.items():
                if k != 'errors':
                    print(f"    [{k}]: {v}")
        if result.metadata.get('errors'):
            print(f"  Warnings: {result.metadata['errors']}")
        if out_path:
            save_result(result, out_path)
    else:
        print(f"  FAILED: {result.error}")
        sys.exit(1)

    return result


def run_cross_arch(stack_a, stack_b, weights_a: Dict[str, torch.Tensor], out_path: str):
    """Cross-arch: weights(A) -> data -> weights(B)."""
    print(f"\nCross-arch: {stack_a} -> {stack_b}")
    print(f"  Input weight keys: {list(weights_a.keys())}")

    result = stack_a.cross_arch(weights_a, stack_b)

    if result.success:
        print(f"  SUCCESS: converted to {len(result.weights)} weight tensors")
        for name, tensor in result.weights.items():
            if isinstance(tensor, torch.Tensor):
                print(f"    {name}: {list(tensor.shape)}")
        if out_path:
            save_result(result, out_path)
    else:
        print(f"  FAILED: {result.error}")
        sys.exit(1)

    return result


def run_key_test(key_name: str):
    """Run a self-test for a key."""
    print(f"\nTesting key: {key_name}")
    key = KEY_REGISTRY[key_name]()

    if key_name == 'linear_mse':
        # Test: generate data, train, compare to key
        torch.manual_seed(42)
        X = torch.randn(20, 3)
        Y = torch.randn(20, 2)

        # Train
        W_train = torch.zeros(2, 3, requires_grad=True)
        opt = torch.optim.Adam([W_train], lr=0.05)
        for _ in range(5000):
            loss = torch.nn.functional.mse_loss(X @ W_train.T, Y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        W_train = W_train.detach()

        # Key
        result = key.forward({'X': X, 'Y': Y})
        W_key = result.weights['W']

        diff = (W_train - W_key).abs().max().item()
        print(f"  Train loss: {torch.nn.functional.mse_loss(X @ W_train.T, Y).item():.8f}")
        print(f"  Key loss:   {torch.nn.functional.mse_loss(X @ W_key.T, Y).item():.8f}")
        print(f"  Max diff:   {diff:.8f}")
        print(f"  PASS: {diff < 1e-3}")

    elif key_name == 'embedding':
        torch.manual_seed(42)
        token_ids = torch.tensor([0, 2, 5, 7])
        targets = torch.randn(4, 4)
        result = key.forward({'token_ids': token_ids, 'target_vectors': targets,
                             'vocab_size': 10, 'd_model': 4})
        W = result.weights['W']
        diff = (W[token_ids] - targets).abs().max().item()
        print(f"  Max diff: {diff:.8f}")
        print(f"  PASS: {diff < 1e-4}")

    elif key_name == 'rmsnorm':
        torch.manual_seed(42)
        X = torch.randn(10, 4)
        true_scale = torch.tensor([2.0, 0.5, 1.0, 3.0])
        eps = 1e-6
        x_norm = X / torch.sqrt(X.pow(2).mean(dim=-1, keepdim=True) + eps)
        Y = x_norm * true_scale
        result = key.forward({'X': X, 'Y': Y, 'eps': eps})
        w = result.weights['weight']
        diff = (w - true_scale).abs().max().item()
        print(f"  True scale: {true_scale.tolist()}")
        print(f"  Key scale:  {w.tolist()}")
        print(f"  Max diff:   {diff:.8f}")
        print(f"  PASS: {diff < 1e-4}")

    else:
        print(f"  (No self-test implemented for {key_name})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="KEYMASTER — orchestrator for keys and keystacks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available keys
  python -m research.keymaster --list-keys

  # Describe the Qwen2 keystack
  python -m research.keymaster --describe qwen2

  # Forward: data -> weights using linear_mse key
  python -m research.keymaster --key linear_mse --forward --data data.pt --out weights.safetensors

  # Test a key
  python -m research.keymaster --key linear_mse --test
        """)

    parser.add_argument('--list-keys', action='store_true', help='List all available keys')
    parser.add_argument('--list-keystacks', action='store_true', help='List all available keystacks')
    parser.add_argument('--describe', type=str, help='Describe a keystack by name')

    parser.add_argument('--key', type=str, help='Use a single key by name')
    parser.add_argument('--keystack', type=str, help='Use a keystack by name')
    parser.add_argument('--target-keystack', type=str, help='Target keystack for cross-arch')

    parser.add_argument('--forward', action='store_true', help='Forward: data -> weights')
    parser.add_argument('--reverse', action='store_true', help='Reverse: weights -> data')
    parser.add_argument('--cross-arch', action='store_true', help='Cross-arch: weights(A) -> weights(B)')
    parser.add_argument('--test', action='store_true', help='Run self-test for a key')

    parser.add_argument('--data', type=str, help='Path to input data file')
    parser.add_argument('--weights', type=str, help='Path to input weights file')
    parser.add_argument('--out', type=str, help='Path to output file')

    args = parser.parse_args()

    # List operations
    if args.list_keys:
        list_keys()
        return
    if args.list_keystacks:
        list_keystacks()
        return
    if args.describe:
        describe_keystack(args.describe)
        return

    # Get the key or keystack
    if args.key:
        if args.key not in KEY_REGISTRY:
            print(f"Error: Key '{args.key}' not found. Use --list-keys to see available keys.")
            sys.exit(1)
        key_or_stack = KEY_REGISTRY[args.key]()
    elif args.keystack:
        if args.keystack not in KEYSTACK_REGISTRY:
            print(f"Error: KeyStack '{args.keystack}' not found. Use --list-keystacks.")
            sys.exit(1)
        key_or_stack = KEYSTACK_REGISTRY[args.keystack]()
    else:
        parser.print_help()
        return

    # Test mode
    if args.test:
        if args.key:
            run_key_test(args.key)
        else:
            print("Test mode requires --key")
        return

    # Forward: data -> weights
    if args.forward:
        if not args.data:
            print("Error: --forward requires --data")
            sys.exit(1)
        data = load_data(args.data)
        run_forward(key_or_stack, data, args.out)
        return

    # Reverse: weights -> data
    if args.reverse:
        if not args.weights:
            print("Error: --reverse requires --weights")
            sys.exit(1)
        weights = load_data(args.weights)
        run_reverse(key_or_stack, weights, args.out)
        return

    # Cross-arch: weights(A) -> weights(B)
    if args.cross_arch:
        if not args.weights or not args.target_keystack:
            print("Error: --cross-arch requires --weights and --target-keystack")
            sys.exit(1)
        weights = load_data(args.weights)
        if args.target_keystack not in KEYSTACK_REGISTRY:
            print(f"Error: target keystack '{args.target_keystack}' not found.")
            sys.exit(1)
        stack_b = KEYSTACK_REGISTRY[args.target_keystack]()
        run_cross_arch(key_or_stack, stack_b, weights, args.out)
        return

    # No operation specified
    print(f"\nSelected: {key_or_stack}")
    if isinstance(key_or_stack, type) and hasattr(key_or_stack, 'describe'):
        print(key_or_stack.describe())
    else:
        print(f"  Name: {key_or_stack.name}")
        print(f"  Class: {key_or_stack.key_class().value}")
        print(f"  Description: {key_or_stack.description}")
    print("\nSpecify an operation: --forward, --reverse, --cross-arch, or --test")


if __name__ == "__main__":
    main()
