"""
Quick test of tiny model loading for debugging.
This script verifies that tiny models can be loaded and used.
"""

import torch
from hig.model.loader import FluxModelLoader


def test_tiny_loader():
    """Test loading tiny components."""
    print("=" * 60)
    print("Testing Tiny Model Loader")
    print("=" * 60)

    # Initialize loader
    loader = FluxModelLoader(
        pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev",  # Not used in tiny mode
        torch_dtype=torch.bfloat16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Load tiny components
    print("\nLoading tiny components...")
    components = loader.load_training_components(use_tiny=True)

    # Verify components
    print("\n" + "=" * 60)
    print("Component Summary")
    print("=" * 60)

    for key, value in components.items():
        if hasattr(value, "config"):
            print(f"\n{key}:")
            if hasattr(value.config, "hidden_size"):
                print(f"  - hidden_size: {value.config.hidden_size}")
            if hasattr(value.config, "d_model"):
                print(f"  - d_model: {value.config.d_model}")
            if hasattr(value.config, "num_layers"):
                print(f"  - num_layers: {value.config.num_layers}")
            if hasattr(value.config, "num_hidden_layers"):
                print(f"  - num_hidden_layers: {value.config.num_hidden_layers}")
        else:
            print(f"\n{key}: {type(value).__name__}")

    # Calculate total parameters
    print("\n" + "=" * 60)
    print("Parameter Counts")
    print("=" * 60)

    total_params = 0
    for key, component in components.items():
        if hasattr(component, "parameters"):
            params = sum(p.numel() for p in component.parameters())
            total_params += params
            print(f"{key}: {params:,} parameters")

    print(f"\nTotal: {total_params:,} parameters")
    print(f"Estimated size: ~{total_params * 2 / 1024 / 1024:.1f} MB (bfloat16)")

    print("\n" + "=" * 60)
    print("Tiny model loading successful!")
    print("=" * 60)


if __name__ == "__main__":
    test_tiny_loader()
