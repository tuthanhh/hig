"""
Flux.1 Model Loader

Loads all components required for Flux.1 training and inference:
- FluxTransformer2DModel (MMDiT architecture)
- FlowMatchEulerDiscreteScheduler
- AutoencoderKL (VAE)
- CLIPTextModel (clip-vit-large-patch14)
- T5EncoderModel (google/t5-v1_1-xxl)
- CLIPTokenizer and T5TokenizerFast
"""

import torch
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    AutoencoderKL,
    FluxTransformer2DModel,
    FluxPipeline,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from typing import Dict, Any, Optional


class FluxModelLoader:
    """
    Loader for Flux.1 model components.

    Supports loading from:
    - black-forest-labs/FLUX.1-dev (full model)
    - black-forest-labs/FLUX.1-schnell (distilled, faster)
    """

    # Default Flux.1 model ID
    DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"

    # Component model IDs (for manual loading)
    CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
    T5_MODEL_ID = "google/t5-v1_1-xxl"

    def __init__(
        self,
        pretrained_model_name_or_path: str = None,
        revision: str = None,
        variant: str = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            pretrained_model_name_or_path: HF Hub ID (e.g. 'black-forest-labs/FLUX.1-dev')
            revision: Model revision branch.
            variant: Model variant (e.g., 'fp16').
            device: 'cuda' or 'cpu'
            torch_dtype: Data type for model weights (bfloat16 recommended for Flux)
        """
        self.model_id = pretrained_model_name_or_path or self.DEFAULT_MODEL_ID
        self.revision = revision
        self.variant = variant
        self.device = device
        self.torch_dtype = torch_dtype

    def _load_tiny_components(self) -> Dict[str, Any]:
        """
        Creates a 'Nano' Flux architecture that is SAFE for memory.

        Fixes VRAM crash by ensuring VAE downsamples image 16x (512->32)
        before running attention.
        """
        print("Loading TINY components (aligned dimensions, deep downsampling)...")

        COMMON_DIM = 32
        HEAD_DIM = 256  # Increased to accommodate RoPE (needs > rope_sum = 8+64+64=136)
        NUM_HEADS = 1  # Reduced to keep total params small
        LATENT_DIM = 16  # Flux uses 16-channel latents, not 4 like SD!

        # 1. Tokenizers
        print("  Loading tokenizers...")
        clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        t5_tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

        # 2. Tiny Text Encoders
        print("  Creating tiny text encoders...")
        from transformers import CLIPTextConfig, T5Config

        clip_config = CLIPTextConfig(
            hidden_size=COMMON_DIM,
            intermediate_size=64,
            projection_dim=COMMON_DIM,
            num_hidden_layers=1,
            num_attention_heads=NUM_HEADS,
            vocab_size=49408,
            max_position_embeddings=77,
        )
        text_encoder_clip = CLIPTextModel(clip_config).to(
            self.device, dtype=self.torch_dtype
        )

        t5_config = T5Config(
            d_model=COMMON_DIM,
            d_ff=64,
            num_layers=1,
            num_heads=NUM_HEADS,
            vocab_size=32128,
        )
        text_encoder_t5 = T5EncoderModel(t5_config).to(
            self.device, dtype=self.torch_dtype
        )

        # 3. Tiny VAE (FIXED)
        print("  Creating tiny VAE...")
        vae = AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=LATENT_DIM,  # This is the KEY parameter - must be 4
            # CRITICAL FIX: Restore 4 downsample blocks.
            # 512x512 -> 256 -> 128 -> 64 -> 32x32.
            down_block_types=(
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
            ),
            up_block_types=(
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
            ),
            # Keep channels minimal to save VRAM/Weights
            block_out_channels=(16, 32, 64, 64),  # Standard progression
            layers_per_block=1,
            norm_num_groups=8,  # Must divide block_out_channels evenly
            act_fn="silu",
            scaling_factor=0.3611,
        ).to(self.device, dtype=self.torch_dtype)

        # 4. Tiny Flux Transformer
        print("  Creating nano Flux transformer...")
        # Flux uses RoPE with axes_dims_rope = (rope_dim, height, width)
        # The attention_head_dim must equal sum(axes_dims_rope) for apply_rotary_emb
        # With 64x64 latents: axes_dims_rope = (8, 64, 64) -> sum = 136
        # So HEAD_DIM must be 136!
        ROPE_DIM = 8
        ROPE_HEIGHT = 64
        ROPE_WIDTH = 64
        REQUIRED_HEAD_DIM = ROPE_DIM + ROPE_HEIGHT + ROPE_WIDTH  # 8 + 64 + 64 = 136

        if HEAD_DIM != REQUIRED_HEAD_DIM:
            print(
                f"  WARNING: Adjusting HEAD_DIM from {HEAD_DIM} to {REQUIRED_HEAD_DIM} for RoPE compatibility"
            )
            HEAD_DIM = REQUIRED_HEAD_DIM

        tiny_flux_config = {
            "patch_size": 1,
            "in_channels": LATENT_DIM,  # 16 channels from VAE
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": HEAD_DIM,  # Must match RoPE sum (136)
            "num_attention_heads": NUM_HEADS,  # 1
            "joint_attention_dim": COMMON_DIM,  # 32
            "pooled_projection_dim": COMMON_DIM,  # 32
            "guidance_embeds": True,
            "axes_dims_rope": (ROPE_DIM, ROPE_HEIGHT, ROPE_WIDTH),  # (8, 64, 64)
        }
        transformer = FluxTransformer2DModel(**tiny_flux_config).to(
            self.device, dtype=self.torch_dtype
        )

        # 5. Scheduler
        scheduler = FlowMatchEulerDiscreteScheduler()

        print("  Tiny components loaded.")

        # Return using consistent naming
        return {
            "scheduler": scheduler,
            "tokenizer_clip": clip_tokenizer,
            "tokenizer_t5": t5_tokenizer,
            "text_encoder_clip": text_encoder_clip,
            "text_encoder_t5": text_encoder_t5,
            "vae": vae,
            "transformer": transformer,
        }

    def load_training_components(
        self,
        load_transformer_in_4bit: bool = False,
        load_text_encoders_in_8bit: bool = False,
        use_tiny: bool = False,
    ) -> Dict[str, Any]:
        """
        Loads individual components for the TRAINING loop.

        Flux.1 architecture requires:
        - transformer: FluxTransformer2DModel (MMDiT)
        - scheduler: FlowMatchEulerDiscreteScheduler
        - vae: AutoencoderKL
        - text_encoder: CLIPTextModel (clip-vit-large-patch14)
        - text_encoder_2: T5EncoderModel (google/t5-v1_1-xxl)
        - tokenizer: CLIPTokenizer
        - tokenizer_2: T5TokenizerFast

        Args:
            load_transformer_in_4bit: Load transformer with 4-bit quantization (saves VRAM)
            load_text_encoders_in_8bit: Load text encoders with 8-bit quantization
            use_tiny: If True, load tiny random models for debugging (<2GB VRAM)
        """
        # Early return for tiny components
        if use_tiny:
            return self._load_tiny_components()

        print(f"FluxLoader: Loading components from {self.model_id}...")

        # 1. Load Scheduler (FlowMatchEulerDiscreteScheduler for Flux)
        print("  Loading scheduler...")
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )

        # 2. Load Tokenizers (Dual tokenizers for CLIP + T5)
        print("  Loading tokenizers...")
        tokenizer_clip = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer", revision=self.revision
        )
        tokenizer_t5 = T5TokenizerFast.from_pretrained(
            self.model_id, subfolder="tokenizer_2", revision=self.revision
        )

        # 3. Load Text Encoders (Dual encoders)
        print("  Loading text encoders...")

        if load_text_encoders_in_8bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            text_encoder_clip = CLIPTextModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                revision=self.revision,
                quantization_config=quantization_config,
                device_map="auto",
            )
            text_encoder_t5 = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder_2",
                revision=self.revision,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            text_encoder_clip = CLIPTextModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                revision=self.revision,
                torch_dtype=self.torch_dtype,
            )
            text_encoder_t5 = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder_2",
                revision=self.revision,
                torch_dtype=self.torch_dtype,
            )

        # 4. Load VAE
        print("  Loading VAE...")
        vae = AutoencoderKL.from_pretrained(
            self.model_id,
            subfolder="vae",
            revision=self.revision,
            torch_dtype=self.torch_dtype,
        )

        # 5. Load Transformer (FluxTransformer2DModel - MMDiT)
        print("  Loading transformer (this may take a while)...")

        if load_transformer_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            )
            transformer = FluxTransformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                revision=self.revision,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            transformer = FluxTransformer2DModel.from_pretrained(
                self.model_id,
                subfolder="transformer",
                revision=self.revision,
                torch_dtype=self.torch_dtype,
            )

        # --- Freeze all components (LoRA will unfreeze specific layers) ---
        transformer.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder_clip.requires_grad_(False)
        text_encoder_t5.requires_grad_(False)

        print("FluxLoader: All components loaded and frozen.")
        print(f"  - Transformer: {type(transformer).__name__}")
        print(f"  - Scheduler: {type(scheduler).__name__}")
        print(f"  - VAE: {type(vae).__name__}")
        print(f"  - Text Encoder CLIP: {type(text_encoder_clip).__name__}")
        print(f"  - Text Encoder T5: {type(text_encoder_t5).__name__}")

        return {
            "transformer": transformer,
            "scheduler": scheduler,
            "vae": vae,
            "text_encoder_clip": text_encoder_clip,
            "text_encoder_t5": text_encoder_t5,
            "tokenizer_clip": tokenizer_clip,
            "tokenizer_t5": tokenizer_t5,
        }

    def load_pipeline_for_inference(
        self, lora_path: Optional[str] = None
    ) -> FluxPipeline:
        """
        Loads a full FluxPipeline for INFERENCE.

        Args:
            lora_path: Optional path to LoRA weights to load.
        """
        print(f"FluxLoader: Loading inference pipeline from {self.model_id}...")

        pipe = FluxPipeline.from_pretrained(
            self.model_id,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
        ).to(self.device)

        if lora_path:
            print(f"FluxLoader: Loading LoRA weights from {lora_path}")
            pipe.load_lora_weights(lora_path)

        # Enable memory optimizations
        pipe.enable_model_cpu_offload()

        print("FluxLoader: Pipeline ready for inference.")
        return pipe


# Backwards compatibility alias
ModelLoader = FluxModelLoader
