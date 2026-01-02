import json
import os
from datasets import Dataset, Features, Value, Image
from tqdm import tqdm
from typing import List, Dict, Optional
from pathlib import Path


# Import the updated translator
from hig.utils.translator import VNTranslator


class DataPreprocessor:
    def __init__(
        self,
        translator: Optional[VNTranslator] = None,
        output_dir: str = "./processed",
        # Arguments to pass to VNTranslator if one isn't provided
        model_path: Optional[str] = None,
        n_gpu_layers: int = -1,
    ):
        """
        Args:
            translator: Existing VNTranslator instance. If None, creates a new one.
            output_dir: Directory to save the processed Hugging Face dataset.
            model_path: Path to GGUF model (passed to VNTranslator if creating new).
            n_gpu_layers: GPU layers for GGUF (passed to VNTranslator if creating new).
        """
        self.output_dir = output_dir

        # Initialize translator if not provided
        if translator:
            self.translator = translator
        else:
            print("Initializing new GGUF Translator for preprocessing...")
            self.translator = VNTranslator(
                model_path=model_path, n_gpu_layers=n_gpu_layers
            )

    def load_jsonl(self, file_path: str) -> List[Dict]:
        """Reads the raw JSONL file line by line."""
        data = []
        print(f"Reading JSONL: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"JSONL file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # Parse each line independently
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
        return data

    def _resolve_image_path(
        self, img_path_str: str, image_root_override: Optional[str] = None
    ) -> Path:
        """
        Resolve image path from string, handling Windows/Linux path conversions.

        Args:
            img_path_str: Original image path string
            image_root_override: Optional root path override

        Returns:
            Resolved Path object
        """
        # Normalize Windows paths to use forward slashes for cross-platform compatibility
        normalized_path = img_path_str.replace("\\", "/")

        # Convert to Path object for cross-platform handling
        img_path = Path(normalized_path)

        # Handle path mapping (e.g., changing drive letters or remapping roots)
        if image_root_override:
            # Use only the filename with the new root
            img_path = Path(image_root_override) / img_path.name
        else:
            # Convert Windows paths to Linux paths
            # E.g., E:/VSCode/hdb/data/02_page_images/... -> data/raw/02_page_images/...
            parts = img_path.parts

            # Handle absolute Windows paths (with drive letters like 'E:')
            if parts and (parts[0].endswith(":") or "\\" in img_path_str):
                # Find 'data' in the path and extract everything after it
                try:
                    data_index = parts.index("data")
                    relative_parts = parts[data_index + 1 :]
                    img_path = Path("data/raw") / Path(*relative_parts)
                except (ValueError, IndexError):
                    # If 'data' not found, keep original path
                    pass

        return img_path

    def process_and_save(
        self,
        jsonl_paths: List[str],
        caption_column: str = "caption_detail",
        image_root_override: Optional[str] = None,
        include_figures: bool = True,
    ):
        """
        Main pipeline: Read JSONL files -> Validate Images -> Translate -> Save Dataset.

        Args:
            jsonl_paths: List of paths to JSONL files to process
            caption_column: Column name for captions (default: "caption_detail")
            image_root_override: Optional root path override for images
            include_figures: Whether to include figure_detail entries (default: True)
        """
        # Ensure jsonl_paths is a list
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]

        valid_images = []
        valid_captions = []

        # Process each JSONL file
        for jsonl_path in jsonl_paths:
            print(f"\n{'=' * 60}")
            print(f"Processing: {jsonl_path}")
            print(f"{'=' * 60}")

            raw_data = self.load_jsonl(jsonl_path)

            # 1. Validation Phase
            print(f"Validating {len(raw_data)} items...")
            for item in tqdm(raw_data, desc="Checking files"):
                # Process main image-caption pair
                img_path_str = item.get("image")
                vn_text = item.get(caption_column)

                if img_path_str and vn_text:
                    img_path = self._resolve_image_path(
                        img_path_str, image_root_override
                    )
                    if img_path.exists():
                        valid_images.append(str(img_path))
                        valid_captions.append(vn_text)

                # Process figure_detail entries if present and enabled
                if include_figures and "figure_detail" in item:
                    figure_details = item.get("figure_detail", [])
                    for figure in figure_details:
                        fig_path_str = figure.get("path")
                        fig_caption = figure.get("caption")

                        if fig_path_str and fig_caption:
                            fig_path = self._resolve_image_path(
                                fig_path_str, image_root_override
                            )
                            if fig_path.exists():
                                valid_images.append(str(fig_path))
                                valid_captions.append(fig_caption)

            print(f"Found {len(valid_images)} total valid pairs so far.")

        print(f"\n{'=' * 60}")
        print(f"Total valid pairs from all files: {len(valid_images)}")
        print(f"{'=' * 60}\n")

        # Check if we have any valid data
        if len(valid_images) == 0:
            print("Warning: No valid image-caption pairs found. Skipping this file.")
            return

        # 2. Translation Phase
        print("Starting translation...")
        en_captions = []

        # For testing, limit to first item
        # valid_captions = [valid_captions[0]]
        # valid_images = [valid_images[0]]

        # We process one by one because GGUF via python loop is safer than batching manually
        # and allows for a nice progress bar.
        for vn_text in tqdm(valid_captions, desc="Translating"):
            try:
                en_text = self.translator.translate(vn_text)
                # Fallback check: if translation returns empty, use original or placeholder
                if not en_text:
                    print("Translation returned empty string, using original text.")
                    en_text = vn_text
                en_captions.append(en_text)
            except Exception as e:
                print(f"Translation failed: {e}")
                en_captions.append(vn_text)

        print("Translation complete. Create a jsonl file to inspect samples:")
        # Save a sample JSONL for inspection

        sample_output_path = os.path.join(self.output_dir, "sample_translations.jsonl")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(sample_output_path, "w", encoding="utf-8") as sample_f:
            for img_path, vn_text, en_text in zip(
                valid_images, valid_captions, en_captions
            ):
                sample_entry = {
                    "image": img_path,
                    "caption_vn": vn_text,
                    "caption_en": en_text,
                }
                sample_f.write(json.dumps(sample_entry, ensure_ascii=False) + "\n")
        print(f"Sample translations saved to: {sample_output_path}")

        # 3. Create Hugging Face Dataset
        print("Constructing Dataset...")
        dataset_dict = {
            "image": valid_images,  # Paths (strings)
            "text": en_captions,  # Translated English text
        }

        hf_dataset = Dataset.from_dict(dataset_dict)

        # Cast 'image' column to Image feature (lazy loading)
        hf_dataset = hf_dataset.cast_column("image", Image())

        # 4. Save to Disk
        os.makedirs(self.output_dir, exist_ok=True)
        hf_dataset.save_to_disk(self.output_dir)
        print(f"Success! Dataset saved to: {os.path.abspath(self.output_dir)}")


# Usage Example
if __name__ == "__main__":
    # Example paths
    raw_data_folder = Path("data/raw/04_output")

    # Initialize
    processor = DataPreprocessor(
        output_dir="data/processed",
        n_gpu_layers=-1,  # Use all GPU layers for translation speed
    )

    # Collect all JSONL files
    jsonl_files = [
        str(raw_data_folder / filename)
        for filename in os.listdir(raw_data_folder)
        if filename.endswith(".jsonl")
    ]

    # Process all files at once
    if jsonl_files:
        print(f"Found {len(jsonl_files)} JSONL files to process")
        processor.process_and_save(jsonl_files)
    else:
        print("No JSONL files found")
