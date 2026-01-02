"""
Flux.1 LoRA Trainer

Implements training loop for Flux.1 with:
- FluxTransformer2DModel (MMDiT) with LoRA
- FlowMatchEulerDiscreteScheduler (Rectified Flow)
- Dual text encoders (CLIP + T5)
- VAE for latent encoding
"""

import torch
import torch.nn.functional as F
import os
from accelerate import Accelerator
from tqdm.auto import tqdm
from diffusers.training_utils import compute_density_for_timestep_sampling
import numpy as np


class FluxTrainer:
    """
    Trainer for Flux.1 LoRA fine-tuning.

    Uses Flow Matching (Rectified Flow) loss:
    - Sample timestep t uniformly from [0, 1]
    - Create noisy latent: x_t = (1-t)*x_0 + t*noise
    - Predict velocity: v = noise - x_0
    - Loss: MSE(predicted_v, target_v)
    """

    def __init__(
        self, components: dict, train_dataloader, accelerator: Accelerator, args
    ):
        """
        Args:
            components: Dict containing Flux model components from FluxModelLoader
            train_dataloader: PyTorch DataLoader with training data
            accelerator: HuggingFace Accelerator for distributed training
            args: Training arguments (learning_rate, num_train_epochs, etc.)
        """
        self.transformer = components["transformer"]
        self.vae = components["vae"]
        self.text_encoder_clip = components["text_encoder_clip"]
        self.text_encoder_t5 = components["text_encoder_t5"]
        self.scheduler = components["scheduler"]

        self.dataloader = train_dataloader
        self.accelerator = accelerator
        self.args = args

        # Flux uses bfloat16 by default
        self.weight_dtype = torch.bfloat16

        # Move frozen models to device (if not using device_map="auto")
        if not hasattr(self.vae, "hf_device_map"):
            self.vae.to(accelerator.device, dtype=self.weight_dtype)
        if not hasattr(self.text_encoder_clip, "hf_device_map"):
            self.text_encoder_clip.to(accelerator.device, dtype=self.weight_dtype)
        if not hasattr(self.text_encoder_t5, "hf_device_map"):
            self.text_encoder_t5.to(accelerator.device, dtype=self.weight_dtype)

    def prepare_optimizer(self):
        """Initialize optimizer for LoRA parameters only."""
        params_to_optimize = list(
            filter(lambda p: p.requires_grad, self.transformer.parameters())
        )

        num_params = sum(p.numel() for p in params_to_optimize)
        print(f"FluxTrainer: Optimizing {num_params:,} parameters")

        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
        )

    def encode_prompt(self, batch):
        """
        Encode text using both CLIP and T5 encoders.

        Flux requires:
        - prompt_embeds: T5 hidden states [batch, seq_len, 4096]
        - pooled_prompt_embeds: CLIP pooled output [batch, 768]
        """
        with torch.no_grad():
            # 1. CLIP Embedding (Pooled output for conditioning)
            clip_input_ids = batch["clip_input_ids"].to(self.accelerator.device)
            clip_outputs = self.text_encoder_clip(
                clip_input_ids, output_hidden_states=False
            )
            pooled_prompt_embeds = clip_outputs.pooler_output  # [batch, 768]

            # 2. T5 Embedding (Sequence for cross-attention)
            t5_input_ids = batch["t5_input_ids"].to(self.accelerator.device)
            t5_outputs = self.text_encoder_t5(t5_input_ids)
            prompt_embeds = t5_outputs.last_hidden_state  # [batch, seq_len, 4096]

        return prompt_embeds, pooled_prompt_embeds

    def encode_images(self, pixel_values):
        """Encode images to latent space using VAE."""
        with torch.no_grad():
            latents = self.vae.encode(
                pixel_values.to(self.vae.dtype)
            ).latent_dist.sample()

            # Apply Flux VAE normalization (if available)
            # Flux.1 VAE uses shift_factor and scaling_factor
            # Standard SD VAE has scaling_factor but shift_factor=None
            shift_factor = getattr(self.vae.config, "shift_factor", None)
            if shift_factor is None:
                shift_factor = 0.0

            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)

            latents = (latents - shift_factor) * scaling_factor

        return latents

    def compute_loss(self, model_pred, noise, latents, timesteps):
        """
        Compute Flow Matching loss.

        For Rectified Flow, the target is the velocity: v = noise - latents
        """
        # Velocity prediction target
        target = noise - latents

        # Weighting can be applied here (e.g., SNR weighting)
        # For now, use simple MSE
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        return loss

    def train(self):
        """Main training loop."""
        # Prepare with accelerator
        self.transformer, self.optimizer, self.dataloader = self.accelerator.prepare(
            self.transformer, self.optimizer, self.dataloader
        )

        num_update_steps = len(self.dataloader) * self.args.num_train_epochs
        max_train_steps = (
            getattr(self.args, "max_train_steps", None) or num_update_steps
        )

        print("***** Starting Flux LoRA Training *****")
        print(f"  Num examples = {len(self.dataloader.dataset)}")
        print(f"  Num epochs = {self.args.num_train_epochs}")
        print(f"  Batch size = {self.dataloader.batch_size}")
        print(f"  Total optimization steps = {num_update_steps}")
        if max_train_steps < num_update_steps:
            print(f"  Early stopping at {max_train_steps} steps")

        global_step = 0

        for epoch in range(self.args.num_train_epochs):
            self.transformer.train()

            progress_bar = tqdm(
                self.dataloader,
                desc=f"Epoch {epoch + 1}/{self.args.num_train_epochs}",
                disable=not self.accelerator.is_local_main_process,
            )

            epoch_loss = 0.0

            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.transformer):
                    # --- 1. Encode Images to Latents ---
                    latents = self.encode_images(batch["pixel_values"])

                    # --- 2. Sample Noise ---
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # --- 3. Sample Timesteps (Flow Matching) ---
                    # Uniform sampling in [0, 1], then scale to scheduler range
                    # Using logit-normal distribution can improve training (optional)
                    u = torch.rand(bsz, device=latents.device, dtype=latents.dtype)

                    # Flux uses sigmoid schedule, scale to [0, 1000]
                    timesteps = (u * 1000).long()

                    # --- 4. Create Noisy Latents (Rectified Flow) ---
                    # x_t = (1 - t) * x_0 + t * noise
                    # The scheduler handles this via add_noise
                    # Move timesteps to CPU for indexing scheduler.sigmas
                    sigmas = self.scheduler.sigmas[timesteps.cpu()].to(latents.device)
                    sigmas = sigmas.view(-1, 1, 1, 1)

                    noisy_latents = (1 - sigmas) * latents + sigmas * noise

                    # --- 5. Encode Text Prompts ---
                    prompt_embeds, pooled_prompt_embeds = self.encode_prompt(batch)

                    # --- 6. Predict Velocity ---
                    # Flux transformer forward pass
                    # Flux expects input in sequence format: [batch, seq_len, channels]
                    # Convert from [batch, channels, height, width] to [batch, height*width, channels]
                    batch_size, channels, height, width = noisy_latents.shape
                    noisy_latents_seq = noisy_latents.permute(0, 2, 3, 1).reshape(
                        batch_size, height * width, channels
                    )

                    # Create position IDs for image and text tokens
                    # img_ids: [height, width, 3] for (batch_id, height_id, width_id)
                    img_ids = torch.zeros(height, width, 3, device=noisy_latents.device)
                    img_ids[..., 1] = (
                        img_ids[..., 1]
                        + torch.arange(height, device=noisy_latents.device)[:, None]
                    )
                    img_ids[..., 2] = (
                        img_ids[..., 2]
                        + torch.arange(width, device=noisy_latents.device)[None, :]
                    )
                    img_ids = img_ids.reshape(height * width, 3)

                    # txt_ids: [text_seq_len, 3] - all zeros for positional encoding
                    txt_seq_len = prompt_embeds.shape[1]
                    txt_ids = torch.zeros(txt_seq_len, 3, device=noisy_latents.device)

                    model_pred = self.transformer(
                        hidden_states=noisy_latents_seq.to(self.weight_dtype),
                        timestep=timesteps.float() / 1000,  # Normalize to [0, 1]
                        guidance=torch.full_like(
                            timesteps.float(), 3.5
                        ),  # Guidance scale for CFG
                        encoder_hidden_states=prompt_embeds.to(self.weight_dtype),
                        pooled_projections=pooled_prompt_embeds.to(self.weight_dtype),
                        img_ids=img_ids,
                        txt_ids=txt_ids,
                        return_dict=False,
                    )[0]

                    # Reshape model_pred back to [batch, channels, height, width]
                    model_pred = model_pred.reshape(
                        batch_size, height, width, channels
                    ).permute(0, 3, 1, 2)

                    # --- 7. Compute Loss ---
                    loss = self.compute_loss(model_pred, noise, latents, timesteps)

                    # --- 8. Backward Pass ---
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.transformer.parameters(), max_norm=1.0
                        )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Update progress
                if self.accelerator.sync_gradients:
                    global_step += 1
                    epoch_loss += loss.detach().item()

                progress_bar.set_postfix(loss=loss.detach().item(), step=global_step)

                # Early stop if max_train_steps reached (for debug mode)
                if global_step >= max_train_steps:
                    print(
                        f"\nReached max_train_steps ({max_train_steps}). Stopping training."
                    )
                    break

            # Break outer loop too
            if global_step >= max_train_steps:
                break

            # End of epoch logging
            avg_epoch_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch {epoch + 1} completed. Average Loss: {avg_epoch_loss:.4f}")

        print("***** Training Complete *****")

    def save_lora(self, output_dir: str):
        """Save the trained LoRA weights."""
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)

            # Unwrap the model and save
            unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
            unwrapped_transformer.save_pretrained(output_dir)

            print(f"FluxTrainer: LoRA weights saved to {output_dir}")


# Backwards compatibility alias
DiffusionTrainer = FluxTrainer
