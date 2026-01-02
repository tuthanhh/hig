import argparse
import os
import sys
import logging

# --- Setup Python Path ---
# This ensures python can find the 'hig' package inside 'src'
# regardless of where you run this script from.
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
sys.path.append(src_path)

from hig.data.preprocessor import DataPreprocessor

"""
Data Preprocessing Script
This script does the following:
1. *Reads a JSONL file* containing image paths and Vietnamese captions.
2. Validates that each image can be opened.
3. Translates Vietnamese captions to English using a GGUF model.
4. Saves the processed dataset in Hugging Face Arrow format.
Basic usage:
uv run scripts/prepare_data.py \
    --jsonl_path data/raw/04_output/metadata.jsonl \
    --output_dir data/processed/dataset \
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset: Validate images, translate captions (VN->EN), and save as Arrow format."
    )

    # Path Arguments
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="data/raw/04_output/metadata.jsonl",
        help="Path to the metadata.jsonl file containing image paths and captions.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/dataset",
        help="Where to save the processed Hugging Face dataset.",
    )

    parser.add_argument(
        "--image_root_override",
        type=str,
        default=None,
        help="Optional: New root folder for images if different from what is in the jsonl file.",
    )

    # Model Arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to local .gguf Qwen model. If None, downloads Qwen3-8B-GGUF automatically.",
    )

    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU. -1 means all layers (fastest).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup simple logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Initializing Data Preprocessor...")

    try:
        # Initialize the processor (this loads the translation model)
        processor = DataPreprocessor(
            output_dir=args.output_dir,
            model_path=args.model_path,
            n_gpu_layers=args.n_gpu_layers,
        )

        # Run the pipeline
        if os.path.exists(args.jsonl_path):
            logger.info(f"Processing file: {args.jsonl_path}")
            processor.process_and_save(
                jsonl_path=args.jsonl_path,
                caption_column="caption_detail",  # Defaults to the detailed caption
                image_root_override=args.image_root_override,
            )
        else:
            logger.error(f"Input file not found: {args.jsonl_path}")

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        raise e


if __name__ == "__main__":
    main()
