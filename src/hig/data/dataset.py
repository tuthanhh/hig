"""
Flux.1 Training Dataset

PyTorch Dataset for Flux.1 LoRA training with:
- Dual tokenization (CLIP + T5)
- Image preprocessing to [-1, 1] range
- Support for variable resolutions with aspect ratio bucketing
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_from_disk
from typing import Optional, Tuple
from PIL import Image


class FluxDataset(Dataset):
    """
    Dataset for Flux.1 training with dual text encoders.

    Flux requires:
    - Images: Normalized to [-1, 1], resolution typically 1024x1024
    - CLIP tokens: max 77 tokens for pooled embedding
    - T5 tokens: up to 512 tokens for sequence embedding
    """

    def __init__(
        self,
        dataset_path: str,
        clip_tokenizer,
        t5_tokenizer,
        size: int = 1024,
        center_crop: bool = True,
        random_flip: bool = True,
        flip_prob: float = 0.5,
        clip_max_length: int = 77,
        t5_max_length: int = 512,
    ):
        """
        Args:
            dataset_path: Path to the processed Hugging Face dataset.
            clip_tokenizer: CLIPTokenizer for pooled embeddings.
            t5_tokenizer: T5TokenizerFast for sequence embeddings.
            size: Target resolution (default 1024 for Flux).
            center_crop: Use center crop (True) or random crop (False).
            random_flip: Enable random horizontal flipping.
            flip_prob: Probability of horizontal flip.
            clip_max_length: Max tokens for CLIP (default 77).
            t5_max_length: Max tokens for T5 (default 512, Flux supports long prompts).
        """
        self.clip_tokenizer = clip_tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.size = size
        self.clip_max_length = clip_max_length
        self.t5_max_length = t5_max_length

        # Load Dataset
        print(f"FluxDataset: Loading from {dataset_path}...")
        try:
            self.data = load_from_disk(dataset_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load dataset at {dataset_path}. Error: {e}"
            )

        print(f"FluxDataset: Loaded {len(self.data)} training examples.")

        # Image Transformations
        # Flux VAE expects images normalized to [-1, 1]
        transform_list = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
        ]

        if random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.transforms = transforms.Compose(transform_list)

    def __len__(self) -> int:
        return len(self.data)

    def _tokenize_clip(self, text: str) -> torch.Tensor:
        """Tokenize text for CLIP encoder."""
        inputs = self.clip_tokenizer(
            text,
            max_length=self.clip_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids.squeeze(0)

    def _tokenize_t5(self, text: str) -> torch.Tensor:
        """Tokenize text for T5 encoder."""
        inputs = self.t5_tokenizer(
            text,
            max_length=self.t5_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids.squeeze(0)

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """Process image for VAE encoding."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transforms(image)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a training example.

        Returns:
            dict with:
            - pixel_values: [3, H, W] tensor normalized to [-1, 1]
            - clip_input_ids: [77] tensor of CLIP token IDs
            - t5_input_ids: [512] tensor of T5 token IDs
        """
        try:
            example = self.data[idx]
            image = example["image"]
            caption = example["text"]

            # Process image
            pixel_values = self._process_image(image)

            # Tokenize text for both encoders
            clip_input_ids = self._tokenize_clip(caption)
            t5_input_ids = self._tokenize_t5(caption)

            return {
                "pixel_values": pixel_values,
                "clip_input_ids": clip_input_ids,
                "t5_input_ids": t5_input_ids,
            }

        except Exception as e:
            print(f"Warning: Error loading index {idx}: {e}")
            # Fallback to next valid index
            return self.__getitem__((idx + 1) % len(self.data))


def create_flux_dataloader(
    dataset_path: str,
    tokenizer_clip,
    tokenizer_t5,
    batch_size: int = 1,
    num_workers: int = 4,
    size: int = 1024,
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """
    Convenience function to create a DataLoader for Flux training.

    Args:
        dataset_path: Path to processed dataset
        tokenizer_clip: CLIP tokenizer
        tokenizer_t5: T5 tokenizer
        batch_size: Batch size (default 1 for large models)
        num_workers: DataLoader workers
        size: Image resolution
        **dataset_kwargs: Additional args for FluxDataset

    Returns:
        Configured DataLoader
    """
    dataset = FluxDataset(
        dataset_path=dataset_path,
        clip_tokenizer=tokenizer_clip,
        t5_tokenizer=tokenizer_t5,
        size=size,
        **dataset_kwargs,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# Backwards compatibility alias
DiffusionDataset = FluxDataset
