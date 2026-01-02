"""
Flux.1 Inference Script

Launch the Vietnamese Historical Image Generator web interface.

Usage:
    # Full mode (inference + training)
    python scripts/run_inference.py [--lora_path path/to/lora] [--share]

    # Training-only mode (no model loading, just training UI)
    python scripts/run_inference.py --training-only [--share]

    # Load with 4-bit quantization (saves ~18GB VRAM)
    python scripts/run_inference.py --quantization 4bit [--share]

    # Load with 8-bit quantization (saves ~12GB VRAM)
    python scripts/run_inference.py --quantization 8bit [--share]
"""

import argparse
import os
import sys

# Add src to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="Launch HIG Web Interface")

    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Flux model ID",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to trained LoRA weights",
    )
    parser.add_argument(
        "--translator_path",
        type=str,
        default=None,
        help="Path to GGUF translator model",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public Gradio share link",
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server hostname",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Launch training-only mode (no model loading)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Load model with quantization (4bit or 8bit). Requires bitsandbytes.",
    )
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Disable CPU offloading (use more VRAM but faster)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HIG - Vietnamese Historical Image Generator")
    print("=" * 60)

    if args.training_only:
        # Training-only mode - no model loading
        from hig.inference.interface import TrainingOnlyInterface

        print("\n🏋️ Launching training-only mode...")
        print("  (No Flux model will be loaded)")

        interface = TrainingOnlyInterface()
        interface.launch(share=args.share, server_name=args.server_name)
    else:
        # Full mode with inference
        from hig.inference.generator import FluxImageGenerator
        from hig.inference.interface import FluxWebInterface

        # Initialize generator
        print("\nInitializing Flux.1 generator...")
        generator = FluxImageGenerator(
            model_id=args.model_id,
            lora_weights_path=args.lora_path,
            translator_model_path=args.translator_path,
            quantization=args.quantization,
            enable_cpu_offload=not args.no_cpu_offload,
        )

        # Launch web interface
        print("\nLaunching web interface...")
        interface = FluxWebInterface(generator)
        interface.launch(share=args.share, server_name=args.server_name)


if __name__ == "__main__":
    main()
