"""
Flux.1 Image Generator

Vietnamese text-to-image generator using Flux.1 pipeline with:
- FluxPipeline for inference
- VNTranslator for Vietnamese to English translation
- Optional LoRA weights for fine-tuned generation
"""

import torch
from diffusers import FluxPipeline
from hig.utils.translator import VNTranslator
from hig.model.loader import FluxModelLoader
from typing import Optional, Tuple
from PIL import Image


class FluxImageGenerator:
    """
    Vietnamese text-to-image generator using Flux.1.

    Workflow:
    1. Translate Vietnamese prompt to English using VNTranslator
    2. Generate image using FluxPipeline
    """

    DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"

    def __init__(
        self,
        model_id: str = None,
        lora_weights_path: str = None,
        translator_model_path: str = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = True,
    ):
        """
        Args:
            model_id: Flux model ID (default: black-forest-labs/FLUX.1-dev)
            lora_weights_path: Path to trained LoRA weights
            translator_model_path: Path to GGUF translation model (None for auto-download)
            device: 'cuda' or 'cpu'
            torch_dtype: Data type for model weights
            enable_cpu_offload: Enable CPU offloading for lower VRAM usage
        """
        self.device = device
        self.torch_dtype = torch_dtype
        self.model_id = model_id or self.DEFAULT_MODEL_ID

        # 1. Initialize Translator
        print("FluxGenerator: Initializing Vietnamese translator...")
        self.translator = VNTranslator(model_path=translator_model_path)

        # 2. Initialize Flux Pipeline
        print(f"FluxGenerator: Loading Flux pipeline from {self.model_id}...")
        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
        )

        # 3. Load LoRA weights if provided
        if lora_weights_path:
            print(f"FluxGenerator: Loading LoRA weights from {lora_weights_path}")
            self.pipe.load_lora_weights(lora_weights_path)

        # 4. Enable memory optimizations
        if enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)

        print("FluxGenerator: Ready for inference.")

    def generate(
        self,
        prompt_vn: str,
        negative_prompt: str = None,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int = -1,
        max_sequence_length: int = 512,
    ) -> Tuple[Image.Image, str]:
        """
        Generate an image from a Vietnamese prompt.

        Args:
            prompt_vn: Vietnamese text prompt
            negative_prompt: Negative prompt (optional, less effective with Flux)
            width: Image width (must be multiple of 16, default 1024)
            height: Image height (must be multiple of 16, default 1024)
            num_inference_steps: Number of denoising steps (default 28 for FLUX.1-dev)
            guidance_scale: Classifier-free guidance scale (default 3.5 for Flux)
            seed: Random seed (-1 for random)
            max_sequence_length: Maximum sequence length for T5 encoder

        Returns:
            Tuple of (generated image, translated English prompt)
        """
        # 1. Translate Vietnamese to English
        print(f"Input (VN): {prompt_vn}")
        prompt_en = self.translator._generate_translation(prompt_vn)
        print(f"Translated (EN): {prompt_en}")

        # 2. Handle seed
        if seed == -1:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # 3. Ensure dimensions are valid for Flux
        width = (width // 16) * 16
        height = (height // 16) * 16

        # 4. Generate image
        print(f"Generating {width}x{height} image with {num_inference_steps} steps...")
        result = self.pipe(
            prompt=prompt_en,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            max_sequence_length=max_sequence_length,
        )

        image = result.images[0]
        print(f"Generation complete. Seed: {seed}")

        return image, prompt_en

    def generate_batch(
        self,
        prompts_vn: list,
        **kwargs,
    ) -> list:
        """
        Generate multiple images from Vietnamese prompts.

        Args:
            prompts_vn: List of Vietnamese prompts
            **kwargs: Additional arguments passed to generate()

        Returns:
            List of (image, translated_prompt) tuples
        """
        results = []
        for prompt in prompts_vn:
            result = self.generate(prompt, **kwargs)
            results.append(result)
        return results


# Backwards compatibility alias
ImageGenerator = FluxImageGenerator
