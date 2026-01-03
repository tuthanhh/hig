"""
Export Dataset Utility

Reads a HuggingFace dataset or JSONL file and exports:
- Images (copied to output folder)
- Caption text files (English captions as .txt files)

Both files share the same name, e.g.:
  image_001.png
  image_001.txt

Usage:
    # From HuggingFace dataset folder
    python scripts/export_dataset.py --input data/processed/dataset --output data/exported

    # From JSONL file
    python scripts/export_dataset.py --input data/processed/dataset/sample_translations.jsonl --output data/exported

    # Specify caption language (default: en)
    python scripts/export_dataset.py --input data/processed/dataset --output data/exported --caption_lang vn
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Literal

from datasets import load_from_disk
from PIL import Image


def export_from_jsonl(
    jsonl_path: str,
    output_dir: str,
    caption_lang: Literal["en", "vn"] = "en",
    base_path: str = None,
) -> int:
    """
    Export images and captions from a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file
        output_dir: Output directory for exported files
        caption_lang: Which caption to use ('en' or 'vn')
        base_path: Base path for resolving relative image paths

    Returns:
        Number of exported items
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine base path for images
    if base_path is None:
        # Try to find the project root (where data/ folder is)
        jsonl_dir = Path(jsonl_path).parent
        # Go up until we find 'data' folder or reach root
        current = jsonl_dir
        while current.parent != current:
            if (current / "data").exists():
                base_path = str(current)
                break
            current = current.parent
        else:
            base_path = str(jsonl_dir)

    caption_key = f"caption_{caption_lang}"
    count = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue

            item = json.loads(line)

            # Get image path and caption
            image_path = item.get("image", "")
            caption = item.get(caption_key, "")

            if not image_path or not caption:
                print(f"Skipping item {idx}: missing image or caption")
                continue

            # Resolve image path
            if not os.path.isabs(image_path):
                full_image_path = os.path.join(base_path, image_path)
            else:
                full_image_path = image_path

            if not os.path.exists(full_image_path):
                print(f"Skipping item {idx}: image not found at {full_image_path}")
                continue

            # Generate output filename
            ext = Path(full_image_path).suffix
            output_name = f"image_{idx:04d}"

            # Copy image
            output_image_path = os.path.join(output_dir, f"{output_name}{ext}")
            shutil.copy2(full_image_path, output_image_path)

            # Write caption
            output_txt_path = os.path.join(output_dir, f"{output_name}.txt")
            with open(output_txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(caption)

            count += 1
            if count % 100 == 0:
                print(f"Exported {count} items...")

    return count


def export_from_hf_dataset(
    dataset_path: str,
    output_dir: str,
    caption_lang: Literal["en", "vn"] = "en",
    image_column: str = "image",
    caption_column: str = None,
) -> int:
    """
    Export images and captions from a HuggingFace dataset.

    Args:
        dataset_path: Path to the HuggingFace dataset folder
        output_dir: Output directory for exported files
        caption_lang: Which caption to use ('en' or 'vn')
        image_column: Name of the image column
        caption_column: Name of the caption column (auto-detected if None)

    Returns:
        Number of exported items
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    # Auto-detect caption column
    if caption_column is None:
        # Try various common column names
        possible_columns = []
        if caption_lang == "en":
            possible_columns = ["caption_en", "text_en", "caption", "text", "prompt"]
        else:
            possible_columns = ["caption_vn", "text_vn", "caption", "text", "prompt"]

        for col in possible_columns:
            if col in dataset.column_names:
                caption_column = col
                break

        if caption_column is None:
            print(
                f"Error: Could not find caption column. Available columns: {dataset.column_names}"
            )
            return 0

    print(f"Using image column: {image_column}")
    print(f"Using caption column: {caption_column}")
    print(f"Dataset size: {len(dataset)}")

    count = 0

    for idx, item in enumerate(dataset):
        try:
            # Get image and caption
            image = item.get(image_column)
            caption = item.get(caption_column, "")

            if image is None or not caption:
                print(f"Skipping item {idx}: missing image or caption")
                continue

            # Generate output filename
            output_name = f"image_{idx:04d}"

            # Save image
            if isinstance(image, Image.Image):
                output_image_path = os.path.join(output_dir, f"{output_name}.png")
                image.save(output_image_path)
            elif isinstance(image, str):
                # Image is a path
                ext = Path(image).suffix or ".png"
                output_image_path = os.path.join(output_dir, f"{output_name}{ext}")
                if os.path.exists(image):
                    shutil.copy2(image, output_image_path)
                else:
                    print(f"Skipping item {idx}: image path not found: {image}")
                    continue
            elif isinstance(image, dict) and "path" in image:
                # Image dict with path
                img_path = image["path"]
                ext = Path(img_path).suffix or ".png"
                output_image_path = os.path.join(output_dir, f"{output_name}{ext}")
                if os.path.exists(img_path):
                    shutil.copy2(img_path, output_image_path)
                else:
                    print(f"Skipping item {idx}: image path not found: {img_path}")
                    continue
            elif isinstance(image, dict) and "bytes" in image:
                # Image dict with bytes
                output_image_path = os.path.join(output_dir, f"{output_name}.png")
                img = Image.open(io.BytesIO(image["bytes"]))
                img.save(output_image_path)
            else:
                print(f"Skipping item {idx}: unknown image format: {type(image)}")
                continue

            # Write caption
            output_txt_path = os.path.join(output_dir, f"{output_name}.txt")
            with open(output_txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(caption)

            count += 1
            if count % 100 == 0:
                print(f"Exported {count} items...")

        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Export dataset images and captions to folder"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to HuggingFace dataset folder or JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for exported files",
    )
    parser.add_argument(
        "--caption_lang",
        type=str,
        choices=["en", "vn"],
        default="en",
        help="Caption language to export (default: en)",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Base path for resolving relative image paths (JSONL only)",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="Name of image column (HuggingFace dataset only)",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="Name of caption column (auto-detected if not specified)",
    )

    args = parser.parse_args()

    input_path = args.input

    print("=" * 60)
    print("Dataset Export Utility")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")
    print(f"Caption language: {args.caption_lang}")
    print("=" * 60)

    # Determine input type
    if input_path.endswith(".jsonl"):
        print("\nDetected JSONL file...")
        count = export_from_jsonl(
            jsonl_path=input_path,
            output_dir=args.output,
            caption_lang=args.caption_lang,
            base_path=args.base_path,
        )
    elif os.path.isdir(input_path):
        # Check if it's a HuggingFace dataset
        if os.path.exists(os.path.join(input_path, "dataset_info.json")):
            print("\nDetected HuggingFace dataset...")
            count = export_from_hf_dataset(
                dataset_path=input_path,
                output_dir=args.output,
                caption_lang=args.caption_lang,
                image_column=args.image_column,
                caption_column=args.caption_column,
            )
        else:
            # Check for JSONL files in the folder
            jsonl_files = list(Path(input_path).glob("*.jsonl"))
            if jsonl_files:
                print(f"\nFound JSONL file: {jsonl_files[0]}")
                count = export_from_jsonl(
                    jsonl_path=str(jsonl_files[0]),
                    output_dir=args.output,
                    caption_lang=args.caption_lang,
                    base_path=args.base_path,
                )
            else:
                print(f"Error: Could not determine input format for {input_path}")
                return
    else:
        print(f"Error: Input path does not exist: {input_path}")
        return

    print("=" * 60)
    print(f"✅ Export complete! {count} items exported to {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    import io  # Import here to avoid issues if not needed

    main()
