import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_from_disk
from typing import Optional


class DiffusionDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        size: int = 512,
        center_crop: bool = False,
        flip_prob: float = 0.5,
        max_token_length: Optional[int] = None,
    ):
        """
        A PyTorch Dataset that loads processed image-text pairs for Diffusion training.

        Args:
            dataset_path (str): Path to the folder containing the saved Hugging Face dataset.
            tokenizer: The CLIPTokenizer instance (loaded in the training script).
            size (int): Target image resolution (e.g., 512 or 768).
            center_crop (bool): If True, crops center. If False, random crops (better for training).
            flip_prob (float): Probability of horizontal flip (0.0 to disable).
            max_token_length (int): Token limit. Defaults to tokenizer's limit (usually 77).
        """
        self.tokenizer = tokenizer
        self.size = size
        self.max_token_length = max_token_length or tokenizer.model_max_length

        # 1. Load Dataset from Disk (Memory-Mapped)
        print(f"Dataset: Loading from {dataset_path}...")
        try:
            self.data = load_from_disk(dataset_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load dataset at {dataset_path}. Error: {e}"
            )

        print(f"Dataset: Found {len(self.data)} training pairs.")

        # 2. Define Image Transformations
        # We normalize to [-1, 1] which is required by the UNet
        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(p=flip_prob),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            example = self.data[idx]
            image = example["image"]
            caption = example["text"]

            # 1. Image Processing
            if not image.mode == "RGB":
                image = image.convert("RGB")

            # Apply transforms (Resize -> Crop -> Flip -> ToTensor -> Normalize)
            pixel_values = self.transforms(image)

            # 2. Text Processing (Tokenization)
            inputs = self.tokenizer(
                caption,
                max_length=self.max_token_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            # Remove the batch dimension added by return_tensors="pt"
            input_ids = inputs.input_ids[0]

            return {"pixel_values": pixel_values, "input_ids": input_ids}

        except Exception as e:
            print(f"Warning: Error loading index {idx}: {e}")
            # Fallback strategy: return the next item (or previous) to prevent crash
            # Simple recursive retry on the next index:
            new_idx = (idx + 1) % len(self.data)
            return self.__getitem__(new_idx)
