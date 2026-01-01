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

    def process_and_save(
        self,
        jsonl_path: str,
        caption_column: str = "caption_detail",
        image_root_override: Optional[str] = None,
    ):
        """
        Main pipeline: Read JSONL -> Validate Images -> Translate -> Save Dataset.
        """
        raw_data = self.load_jsonl(jsonl_path)

        valid_images = []
        valid_captions = []

        # 1. Validation Phase
        print(f"Validating {len(raw_data)} items...")
        for item in tqdm(raw_data, desc="Checking files"):
            img_path = item.get("image")
            vn_text = item.get(caption_column)

            if not img_path or not vn_text:
                continue

            # Handle path mapping (e.g., changing drive letters)
            if image_root_override:
                # Example: If raw path is "E:\data\img.png" and override is "/data"
                # We strip the folder and join with new root.
                # Adjust this logic based on your specific folder structure needs.
                filename = os.path.basename(img_path)
                # OR: relative_path = os.path.relpath(img_path, "OLD_ROOT")
                img_path = os.path.join(image_root_override, filename)
            else:
                # Convert Windows paths to Linux paths
                # E.g., E:\VSCode\hdb\data\02_page_images\... -> data/raw/02_page_images/...
                if "\\" in img_path or img_path.startswith("E:"):
                    # Extract the path after "data\"
                    parts = img_path.replace("\\", "/").split("/")
                    if "data" in parts:
                        data_index = parts.index("data")
                        relative_path = "/".join(parts[data_index + 1 :])
                        img_path = os.path.join("data/raw", relative_path)

            if os.path.exists(img_path):
                valid_images.append(img_path)
                valid_captions.append(vn_text)
            else:
                # Optional: print only first few errors to avoid spam
                pass

        print(f"Found {len(valid_images)} valid pairs out of {len(raw_data)} total.")

        # Check if we have any valid data
        if len(valid_images) == 0:
            print("Warning: No valid image-caption pairs found. Skipping this file.")
            return

        # 2. Translation Phase
        print("Starting translation...")
        en_captions = []

        # We process one by one because GGUF via python loop is safer than batching manually
        # and allows for a nice progress bar.
        for vn_text in tqdm(valid_captions, desc="Translating"):
            try:
                en_text = self.translator.translate(vn_text)
                # Fallback check: if translation returns empty, use original or placeholder
                if not en_text:
                    en_text = vn_text
                en_captions.append(en_text)
            except Exception as e:
                print(f"Translation failed: {e}")
                en_captions.append(vn_text)

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
        print(f"✅ Success! Dataset saved to: {os.path.abspath(self.output_dir)}")


# Usage Example
if __name__ == "__main__":
    # Example paths
    raw_data_folder = Path("data/raw/04_output")

    # Initialize
    processor = DataPreprocessor(
        output_dir="data/processed",
        n_gpu_layers=-1,  # Use all GPU layers for translation speed
    )

    # Run
    for filename in os.listdir(raw_data_folder):
        if filename.endswith(".jsonl"):
            jsonl_path = os.path.join(raw_data_folder, filename)
            processor.process_and_save(jsonl_path)
