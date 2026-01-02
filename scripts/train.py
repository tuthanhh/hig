"""
Flux.1 LoRA Training Script

Train a LoRA adapter on Flux.1 for Vietnamese historical image generation.

Usage:
    # Production training
    python -m hig.train --dataset_path data/processed --output_dir output/lora

    # Tiny model debug mode (random init, <2GB VRAM)
    python -m hig.train --dataset_path data/processed --tiny --max_steps 10

    # Custom training with all options
    python -m hig.train \
        --dataset_path data/processed \
        --output_dir output/lora \
        --model_id black-forest-labs/FLUX.1-dev \
        --lora_rank 16 \
        --lora_alpha 16 \
        --epochs 10 \
        --lr 1e-4 \
        --batch_size 1 \
        --grad_accum 4 \
        --resolution 1024 \
        --save_steps 500 \
        --mixed_precision bf16

Requirements:
    - GPU with at least 24GB VRAM for full model (4-bit quantization)
    - Processed dataset from preprocessor.py
    - HuggingFace access to black-forest-labs/FLUX.1-dev
"""

import argparse
import logging
import os
import sys
import math
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from dataclasses import dataclass

# Add src to path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@dataclass
class TrainingArgs:
    """Training arguments for Flux.1 LoRA fine-tuning."""

    # Paths
    dataset_path: str
    output_dir: str
    model_id: str

    # Model configuration
    use_tiny: bool
    load_transformer_in_4bit: bool
    load_text_encoders_in_8bit: bool

    # LoRA configuration
    lora_rank: int
    lora_alpha: int

    # Training configuration
    num_train_epochs: int
    max_train_steps: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    resolution: int

    # Optimizer settings
    use_8bit_adam: bool
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: float

    # Scheduler settings
    lr_scheduler: str
    lr_warmup_steps: int

    # Training enhancements
    snr_gamma: float
    noise_offset: float

    # Checkpointing
    save_steps: int

    # Misc
    seed: int
    center_crop: bool


def train(args: TrainingArgs):
    """Main training function."""

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("hig - Flux.1 LoRA Training")
    if args.use_tiny:
        print(" TINY MODE (random init models for debugging)")
    print("=" * 60)

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    logger.info(accelerator.state)

    # Set seed for reproducibility
    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        set_seed(args.seed)

    # Import here to avoid loading models before accelerator is ready
    from hig.model.loader import FluxModelLoader
    from hig.model.adapter import FluxLoraAdapter
    from hig.trainer import FluxTrainer
    from hig.data.dataset import create_flux_dataloader

    # 1. Load Model Components
    print("\n[1/5] Loading Flux.1 model components...")
    if not args.use_tiny:
        print(f"  Model: {args.model_id}")

    loader = FluxModelLoader(
        pretrained_model_name_or_path=args.model_id,
        torch_dtype=torch.bfloat16,
        device=accelerator.device,
    )

    components = loader.load_training_components(
        load_transformer_in_4bit=args.load_transformer_in_4bit,
        load_text_encoders_in_8bit=args.load_text_encoders_in_8bit,
        use_tiny=args.use_tiny,
    )

    # 2. Apply LoRA
    print("\n[2/5] Applying LoRA adapter...")
    print(f"  Rank: {args.lora_rank}, Alpha: {args.lora_alpha}")
    adapter = FluxLoraAdapter(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    components["transformer"] = adapter.apply(components["transformer"])

    # 3. Create DataLoader
    print("\n[3/5] Creating DataLoader...")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")

    train_dataloader = create_flux_dataloader(
        dataset_path=args.dataset_path,
        tokenizer_clip=components["tokenizer_clip"],
        tokenizer_t5=components["tokenizer_t5"],
        batch_size=args.batch_size,
        size=args.resolution,
    )

    # 4. Calculate training steps
    if args.max_train_steps is None:
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps
        )
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        logger.info(f"Calculated max_train_steps: {args.max_train_steps}")

    # 5. Initialize Trainer
    print("\n[4/5] Initializing trainer...")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_train_epochs}")
    print(f"  Max steps: {args.max_train_steps}")
    if args.snr_gamma:
        print(f"  SNR gamma: {args.snr_gamma}")
    if args.noise_offset:
        print(f"  Noise offset: {args.noise_offset}")

    trainer = FluxTrainer(
        components=components,
        train_dataloader=train_dataloader,
        accelerator=accelerator,
        args=args,
    )
    trainer.prepare_optimizer()

    # 6. Train
    print("\n[5/5] Starting training...")
    trainer.train()

    # 7. Save final LoRA weights
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        trainer.save_lora(args.output_dir)
        logger.info(f"Training complete! LoRA saved to: {args.output_dir}")

    print("\n" + "=" * 60)
    print(f"Training complete! LoRA saved to: {args.output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train Flux.1 LoRA for Vietnamese Historical Image Generation"
    )

    # === Paths ===
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/processed",
        help="Path to processed dataset (from preprocessor.py)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/lora",
        help="Output directory for LoRA weights and checkpoints",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Flux model ID (ignored if --tiny is used)",
    )

    # === Model Configuration ===
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use tiny random models for debugging (<2GB VRAM, no pretrained weights)",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="Disable 4-bit quantization for transformer",
    )
    parser.add_argument(
        "--no_8bit",
        action="store_true",
        help="Disable 8-bit quantization for text encoders",
    )

    # === LoRA Configuration ===
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank (dimension, higher = more capacity)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (scaling factor)",
    )

    # === Training Configuration ===
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max training steps (overrides epochs if set, useful for debugging)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Training resolution (512 or 1024 recommended)",
    )

    # === Optimizer Settings ===
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use BitsAndBytes 8-bit Adam to save VRAM",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1 parameter",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 parameter",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Adam weight decay",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Adam epsilon parameter",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )

    # === Scheduler Settings ===
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps for learning rate scheduler",
    )

    # === Training Enhancements ===
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="Min-SNR weighting gamma for better detail (try 5.0)",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.0,
        help="Noise offset for better contrast (try 0.1)",
    )

    # === Checkpointing ===
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )

    # === Misc ===
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        default=True,
        help="Center crop images during preprocessing",
    )

    args_parsed = parser.parse_args()

    # Create TrainingArgs
    args = TrainingArgs(
        # Paths
        dataset_path=args_parsed.dataset_path,
        output_dir=args_parsed.output_dir,
        model_id=args_parsed.model_id,
        # Model configuration
        use_tiny=args_parsed.tiny,
        load_transformer_in_4bit=not args_parsed.no_4bit,
        load_text_encoders_in_8bit=not args_parsed.no_8bit,
        # LoRA configuration
        lora_rank=args_parsed.lora_rank,
        lora_alpha=args_parsed.lora_alpha,
        # Training configuration
        num_train_epochs=args_parsed.epochs,
        max_train_steps=args_parsed.max_steps,
        learning_rate=args_parsed.lr,
        batch_size=args_parsed.batch_size,
        gradient_accumulation_steps=args_parsed.grad_accum,
        resolution=args_parsed.resolution,
        # Optimizer settings
        use_8bit_adam=args_parsed.use_8bit_adam,
        adam_beta1=args_parsed.adam_beta1,
        adam_beta2=args_parsed.adam_beta2,
        adam_weight_decay=args_parsed.adam_weight_decay,
        adam_epsilon=args_parsed.adam_epsilon,
        max_grad_norm=args_parsed.max_grad_norm,
        # Scheduler settings
        lr_scheduler=args_parsed.lr_scheduler,
        lr_warmup_steps=args_parsed.lr_warmup_steps,
        # Training enhancements
        snr_gamma=args_parsed.snr_gamma,
        noise_offset=args_parsed.noise_offset,
        # Checkpointing
        save_steps=args_parsed.save_steps,
        # Misc
        seed=args_parsed.seed,
        center_crop=args_parsed.center_crop,
    )

    train(args)


if __name__ == "__main__":
    main()
