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
1. *Reads JSONL files* containing image paths and Vietnamese captions.
2. Validates that each image can be opened.
3. Translates Vietnamese captions to English using a GGUF model.
4. Saves the processed dataset in Hugging Face Arrow format.

Basic usage (single file):
    uv run scripts/prepare_data.py \
        --jsonl_paths data/raw/04_output/metadata.jsonl \
        --output_dir data/processed/dataset

Basic usage (multiple files):
    uv run scripts/prepare_data.py \
        --jsonl_paths data/raw/04_output/file1.jsonl data/raw/04_output/file2.jsonl \
        --output_dir data/processed/dataset

Basic usage (directory - processes all .jsonl files):
    uv run scripts/prepare_data.py \
        --jsonl_dir data/raw/04_output \
        --output_dir data/processed/dataset
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess dataset: Validate images, translate captions (VN->EN), and save as Arrow format."
    )

    # Path Arguments - mutually exclusive group for file/directory input
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--jsonl_paths",
        type=str,
        nargs="+",
        help="Path(s) to JSONL file(s) containing image paths and captions. Can specify multiple files.",
    )

    input_group.add_argument(
        "--jsonl_dir",
        type=str,
        help="Directory containing JSONL files. All .jsonl files in this directory will be processed.",
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

    parser.add_argument(
        "--include_figures",
        action="store_true",
        default=True,
        help="Include figure_detail entries from JSONL files (default: True).",
    )

    parser.add_argument(
        "--no_figures",
        action="store_false",
        dest="include_figures",
        help="Exclude figure_detail entries from JSONL files.",
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
        # Determine which JSONL files to process
        jsonl_paths = []

        if args.jsonl_paths:
            # Direct file paths provided
            jsonl_paths = args.jsonl_paths
            logger.info(f"Processing {len(jsonl_paths)} specified file(s)")

        elif args.jsonl_dir:
            # Directory provided - find all .jsonl files
            jsonl_dir = args.jsonl_dir
            if not os.path.isdir(jsonl_dir):
                logger.error(f"Directory not found: {jsonl_dir}")
                return

            jsonl_paths = [
                os.path.join(jsonl_dir, f)
                for f in os.listdir(jsonl_dir)
                if f.endswith(".jsonl")
            ]
            logger.info(f"Found {len(jsonl_paths)} JSONL file(s) in {jsonl_dir}")

        if not jsonl_paths:
            logger.error("No JSONL files found to process")
            return

        # Validate that all files exist
        missing_files = [p for p in jsonl_paths if not os.path.exists(p)]
        if missing_files:
            logger.error(f"The following files were not found:")
            for f in missing_files:
                logger.error(f"  - {f}")
            return

        # Initialize the processor (this loads the translation model)
        processor = DataPreprocessor(
            output_dir=args.output_dir,
            model_path=args.model_path,
            n_gpu_layers=args.n_gpu_layers,
        )

        # Run the pipeline with all files
        logger.info("Starting processing pipeline...")
        processor.process_and_save(
            jsonl_paths=jsonl_paths,
            caption_column="caption_detail",  # Defaults to the detailed caption
            image_root_override=args.image_root_override,
            include_figures=args.include_figures,
        )

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        raise e


if __name__ == "__main__":
    main()
