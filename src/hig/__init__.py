"""
HIG - Vietnamese Historical Image Generator

A Flux.1-based text-to-image generation system for Vietnamese historical content.

Components:
- FluxModelLoader: Load Flux.1 model components with optional tiny models for debugging
- FluxLoraAdapter: Apply LoRA for fine-tuning
- FluxTrainer: Training loop for LoRA fine-tuning
- FluxDataset: Dataset for training
- FluxImageGenerator: Inference with Vietnamese translation
- FluxWebInterface: Gradio web UI
"""

from hig.model.loader import FluxModelLoader, ModelLoader
from hig.model.adapter import FluxLoraAdapter, LoraAdapter
from hig.trainer import FluxTrainer, DiffusionTrainer
from hig.data.dataset import FluxDataset, DiffusionDataset, create_flux_dataloader
from hig.inference.generator import FluxImageGenerator, ImageGenerator
from hig.inference.interface import FluxWebInterface, WebInterface

__version__ = "0.1.0"
__all__ = [
    # Primary classes
    "FluxModelLoader",
    "FluxLoraAdapter",
    "FluxTrainer",
    "FluxDataset",
    "FluxImageGenerator",
    "FluxWebInterface",
    # Utility functions
    "create_flux_dataloader",
    # Backwards compatibility aliases
    "ModelLoader",
    "LoraAdapter",
    "DiffusionTrainer",
    "DiffusionDataset",
    "ImageGenerator",
    "WebInterface",
]


def main() -> None:
    """CLI entry point."""
    print("HIG - Vietnamese Historical Image Generator")
    print("Use 'python -m hig.train' to start training")
    print("Use 'python -m hig.train --debug' for debug mode (faster)")
    print("Use 'python -m hig.inference' to start the web interface")
