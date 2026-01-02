"""
LoRA Adapter for Flux.1 Transformer

Implements Parameter-Efficient Fine-Tuning (PEFT) using LoRA
specifically optimized for FluxTransformer2DModel (MMDiT architecture).
"""

from peft import LoraConfig, get_peft_model
from diffusers import FluxTransformer2DModel


class FluxLoraAdapter:
    """
    LoRA adapter specifically designed for Flux.1's MMDiT architecture.

    Target modules for FluxTransformer2DModel:
    - Attention projections: to_q, to_k, to_v, to_out (in transformer blocks)
    - Joint attention blocks (for image-text cross attention)
    - Single transformer blocks
    """

    # Default target modules for FluxTransformer2DModel
    # These target the attention layers in the MMDiT blocks
    FLUX_TARGET_MODULES = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        # Additional modules for better adaptation
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
        # Feed-forward layers (optional, increases params but can improve quality)
        # "ff.net.0.proj",
        # "ff.net.2",
    ]

    def __init__(
        self,
        rank: int = 16,
        alpha: int = 16,
        target_modules: list = None,
        dropout: float = 0.0,
    ):
        """
        Args:
            rank: The dimension of the LoRA update matrices (higher = more capacity).
            alpha: The scaling factor (alpha/rank determines effective learning rate).
            target_modules: Which layers to apply LoRA to. None uses FLUX_TARGET_MODULES.
            dropout: Dropout probability for LoRA layers.
        """
        if target_modules is None:
            target_modules = self.FLUX_TARGET_MODULES

        self.config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
            task_type=None,  # No specific task type for diffusion models
        )
        self.rank = rank
        self.alpha = alpha

    def apply(self, transformer: FluxTransformer2DModel) -> FluxTransformer2DModel:
        """
        Injects LoRA layers into the Flux transformer.

        The transformer should be frozen before calling this method.
        After injection, only LoRA parameters will be trainable.

        Args:
            transformer: FluxTransformer2DModel instance

        Returns:
            PEFT-wrapped transformer with trainable LoRA layers
        """
        print(f"FluxLoraAdapter: Injecting LoRA (r={self.rank}, alpha={self.alpha})...")
        print(f"  Target modules: {self.config.target_modules}")

        # Wrap the transformer with PEFT
        transformer = get_peft_model(transformer, self.config)

        # Print parameter statistics
        trainable_params = sum(
            p.numel() for p in transformer.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in transformer.parameters())

        print("FluxLoraAdapter: LoRA injected successfully.")
        print(f"   - Trainable Params: {trainable_params:,}")
        print(f"   - Total Params:     {total_params:,}")
        print(f"   - % Trainable:      {(trainable_params / total_params) * 100:.4f}%")
        print(
            f"   - VRAM for LoRA:    ~{trainable_params * 4 / 1024 / 1024:.2f} MB (fp32)"
        )

        return transformer


# Backwards compatibility alias
LoraAdapter = FluxLoraAdapter
