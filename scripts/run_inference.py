"""
Flux.1 Inference Script

Launch the Vietnamese Historical Image Generator web interface.

Usage:
    python scripts/run_inference.py [--lora_path path/to/lora] [--share]
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

    args = parser.parse_args()

    # Import here to avoid slow startup
    from hig.inference.generator import FluxImageGenerator
    from hig.inference.interface import FluxWebInterface

    print("=" * 60)
    print("HIG - Vietnamese Historical Image Generator")
    print("=" * 60)

    # Initialize generator
    print("\nInitializing Flux.1 generator...")
    generator = FluxImageGenerator(
        model_id=args.model_id,
        lora_weights_path=args.lora_path,
        translator_model_path=args.translator_path,
    )

    # Launch web interface
    print("\nLaunching web interface...")
    interface = FluxWebInterface(generator)
    interface.launch(share=args.share, server_name=args.server_name)


if __name__ == "__main__":
    main()
