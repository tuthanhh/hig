"""
Gradio Web Interface for Flux.1 Vietnamese Image Generator

Provides an interactive UI for:
- Vietnamese text input with automatic translation
- Image generation with adjustable parameters
- Multiple resolution options optimized for Flux
- Training configuration and launch
"""

import gradio as gr
import subprocess
import threading
import os
from typing import Optional
from hig.inference.generator import FluxImageGenerator


class FluxWebInterface:
    """
    Gradio-based web interface for Vietnamese text-to-image generation and training.
    """

    # Flux-optimized resolution presets
    RESOLUTION_PRESETS = {
        "Square (1024x1024)": (1024, 1024),
        "Portrait (768x1344)": (768, 1344),
        "Landscape (1344x768)": (1344, 768),
        "Wide (1536x640)": (1536, 640),
        "Tall (640x1536)": (640, 1536),
        "HD (1280x720)": (1280, 720),
        "Square Small (512x512)": (512, 512),
    }

    def __init__(self, generator: Optional[FluxImageGenerator] = None):
        """
        Args:
            generator: Initialized FluxImageGenerator instance (optional for training-only mode)
        """
        self.generator = generator
        self.training_process = None
        self.training_log = []

    def predict(
        self,
        prompt: str,
        negative_prompt: str,
        resolution: str,
        steps: int,
        guidance: float,
        seed: int,
        max_sequence_length: int,
    ):
        """
        Generate image from Vietnamese prompt.
        """
        if self.generator is None:
            return None, "Generator not initialized. Please load a model first."

        width, height = self.RESOLUTION_PRESETS[resolution]

        image, translated_text = self.generator.generate(
            prompt_vn=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=int(seed),
            max_sequence_length=int(max_sequence_length),
        )
        return image, translated_text

    def build_training_command(
        self,
        dataset_path: str,
        output_dir: str,
        model_id: str,
        use_tiny: bool,
        lora_rank: int,
        lora_alpha: int,
        epochs: int,
        max_steps: int,
        learning_rate: float,
        batch_size: int,
        grad_accum: int,
        resolution: int,
        save_steps: int,
        use_8bit_adam: bool,
        lr_scheduler: str,
        warmup_steps: int,
        snr_gamma: float,
        noise_offset: float,
        seed: int,
    ) -> str:
        """Build the training command from parameters."""
        cmd = ["uv", "run", "scripts/train.py"]

        # Required paths
        cmd.extend(["--dataset_path", dataset_path])
        cmd.extend(["--output_dir", output_dir])

        # Model selection
        if use_tiny:
            cmd.append("--tiny")
        else:
            cmd.extend(["--model_id", model_id])

        # LoRA config
        cmd.extend(["--lora_rank", str(lora_rank)])
        cmd.extend(["--lora_alpha", str(lora_alpha)])

        # Training config
        cmd.extend(["--epochs", str(epochs)])
        if max_steps > 0:
            cmd.extend(["--max_steps", str(max_steps)])
        cmd.extend(["--lr", str(learning_rate)])
        cmd.extend(["--batch_size", str(batch_size)])
        cmd.extend(["--grad_accum", str(grad_accum)])
        cmd.extend(["--resolution", str(resolution)])
        cmd.extend(["--save_steps", str(save_steps)])

        # Optimizer
        if use_8bit_adam:
            cmd.append("--use_8bit_adam")

        # Scheduler
        cmd.extend(["--lr_scheduler", lr_scheduler])
        cmd.extend(["--warmup_steps", str(warmup_steps)])

        # Training enhancements
        if snr_gamma > 0:
            cmd.extend(["--snr_gamma", str(snr_gamma)])
        if noise_offset > 0:
            cmd.extend(["--noise_offset", str(noise_offset)])

        # Misc
        cmd.extend(["--seed", str(seed)])

        return " ".join(cmd)

    def start_training(
        self,
        dataset_path: str,
        output_dir: str,
        model_id: str,
        use_tiny: bool,
        lora_rank: int,
        lora_alpha: int,
        epochs: int,
        max_steps: int,
        learning_rate: float,
        batch_size: int,
        grad_accum: int,
        resolution: int,
        save_steps: int,
        use_8bit_adam: bool,
        lr_scheduler: str,
        warmup_steps: int,
        snr_gamma: float,
        noise_offset: float,
        seed: int,
    ):
        """Start training in a subprocess."""
        if self.training_process is not None and self.training_process.poll() is None:
            return "⚠️ Training already in progress. Please wait or stop the current training."

        # Validate paths
        if not dataset_path or not os.path.exists(dataset_path):
            return f"❌ Dataset path does not exist: {dataset_path}"

        cmd = self.build_training_command(
            dataset_path,
            output_dir,
            model_id,
            use_tiny,
            lora_rank,
            lora_alpha,
            epochs,
            max_steps,
            learning_rate,
            batch_size,
            grad_accum,
            resolution,
            save_steps,
            use_8bit_adam,
            lr_scheduler,
            warmup_steps,
            snr_gamma,
            noise_offset,
            seed,
        )

        self.training_log = [
            f"🚀 Starting training...\n",
            f"Command: {cmd}\n",
            "=" * 50 + "\n",
        ]

        try:
            self.training_process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # Start thread to read output
            def read_output():
                for line in iter(self.training_process.stdout.readline, ""):
                    self.training_log.append(line)
                    if len(self.training_log) > 500:  # Keep last 500 lines
                        self.training_log = self.training_log[-500:]
                self.training_log.append("\n✅ Training process finished.\n")

            thread = threading.Thread(target=read_output, daemon=True)
            thread.start()

            return f"✅ Training started!\n\nCommand:\n```\n{cmd}\n```"
        except Exception as e:
            return f"❌ Failed to start training: {str(e)}"

    def stop_training(self):
        """Stop the current training process."""
        if self.training_process is None:
            return "No training process running."

        if self.training_process.poll() is None:
            self.training_process.terminate()
            self.training_log.append("\n⛔ Training stopped by user.\n")
            return "⛔ Training stopped."
        else:
            return "Training already finished."

    def get_training_log(self):
        """Get current training log."""
        return (
            "".join(self.training_log) if self.training_log else "No training logs yet."
        )

    def get_training_status(self):
        """Get training status."""
        if self.training_process is None:
            return "⚪ No training started"
        elif self.training_process.poll() is None:
            return "🟢 Training in progress..."
        elif self.training_process.returncode == 0:
            return "✅ Training completed successfully"
        else:
            return f"🔴 Training failed (exit code: {self.training_process.returncode})"

    def launch(self, share: bool = False, server_name: str = "0.0.0.0"):
        """
        Launch the Gradio web interface.

        Args:
            share: Create public share link
            server_name: Server hostname
        """
        with gr.Blocks(
            title="HIG - Vietnamese Historical Image Generator",
            theme=gr.themes.Soft(),
        ) as demo:
            gr.Markdown(
                """
                # 🎨 Vietnamese Historical Image Generator
                ### Powered by Flux.1 + Custom LoRA
                """
            )

            with gr.Tabs():
                # ==================== INFERENCE TAB ====================
                with gr.TabItem("🖼️ Generate Images"):
                    gr.Markdown(
                        """
                        Type a prompt in Vietnamese describing a historical scene,
                        and the AI will translate it and generate an image.
                        """
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            # Input section
                            prompt = gr.Textbox(
                                label="📝 Vietnamese Prompt",
                                placeholder="Ví dụ: Vua Lê Đại Hành cưỡi ngựa ra trận đánh giặc Tống...",
                                lines=3,
                            )

                            negative_prompt = gr.Textbox(
                                label="🚫 Negative Prompt (English, optional)",
                                placeholder="e.g., blurry, low quality, distorted...",
                                lines=2,
                                info="What to avoid in the image (less effective with Flux)",
                            )

                            resolution = gr.Dropdown(
                                label="📐 Resolution",
                                choices=list(self.RESOLUTION_PRESETS.keys()),
                                value="Square (1024x1024)",
                            )

                            with gr.Accordion("⚙️ Advanced Settings", open=False):
                                steps = gr.Slider(
                                    label="Inference Steps",
                                    minimum=10,
                                    maximum=50,
                                    value=28,
                                    step=1,
                                    info="More steps = better quality but slower",
                                )
                                guidance = gr.Slider(
                                    label="Guidance Scale",
                                    minimum=1.0,
                                    maximum=10.0,
                                    value=3.5,
                                    step=0.5,
                                    info="How closely to follow the prompt (3.5 is default for Flux)",
                                )
                                max_sequence_length = gr.Slider(
                                    label="Max Sequence Length",
                                    minimum=128,
                                    maximum=512,
                                    value=512,
                                    step=64,
                                    info="Max tokens for T5 encoder (longer prompts need more)",
                                )
                                seed = gr.Number(
                                    label="Seed",
                                    value=-1,
                                    info="-1 for random seed",
                                )

                            generate_btn = gr.Button(
                                "🎨 Generate Image",
                                variant="primary",
                                size="lg",
                            )

                        with gr.Column(scale=1):
                            # Output section
                            output_image = gr.Image(
                                label="Generated Image",
                                type="pil",
                            )
                            translated_output = gr.Textbox(
                                label="🔄 Translated Prompt (English)",
                                interactive=False,
                            )

                    # Event handler
                    generate_btn.click(
                        fn=self.predict,
                        inputs=[
                            prompt,
                            negative_prompt,
                            resolution,
                            steps,
                            guidance,
                            seed,
                            max_sequence_length,
                        ],
                        outputs=[output_image, translated_output],
                    )

                    # Example prompts
                    gr.Examples(
                        examples=[
                            [
                                "Vua Lê Đại Hành trong bộ áo long bào, đứng trước quân đội"
                            ],
                            ["Một trận thủy chiến trên sông Bạch Đằng với cọc gỗ"],
                            ["Cảnh chợ quê Việt Nam thời xưa với người bán hàng"],
                            ["Kinh thành Thăng Long với cung điện và thành lũy"],
                        ],
                        inputs=prompt,
                    )

                # ==================== TRAINING TAB ====================
                with gr.TabItem("🏋️ Training"):
                    gr.Markdown(
                        """
                        ### Configure and launch LoRA training for Flux.1
                        Set your training parameters below and start training.
                        """
                    )

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### 📂 Paths")
                            train_dataset_path = gr.Textbox(
                                label="Dataset Path",
                                value="data/processed/dataset",
                                info="Path to processed HuggingFace dataset",
                            )
                            train_output_dir = gr.Textbox(
                                label="Output Directory",
                                value="output/lora",
                                info="Where to save LoRA weights and logs",
                            )

                            gr.Markdown("#### 🤖 Model")
                            train_model_id = gr.Textbox(
                                label="Model ID",
                                value="black-forest-labs/FLUX.1-dev",
                                info="HuggingFace model ID",
                            )
                            train_use_tiny = gr.Checkbox(
                                label="Use Tiny Model (Debug Mode)",
                                value=False,
                                info="Use tiny random-init model for testing (<2GB VRAM)",
                            )

                            gr.Markdown("#### 🔧 LoRA Configuration")
                            with gr.Row():
                                train_lora_rank = gr.Slider(
                                    label="LoRA Rank",
                                    minimum=4,
                                    maximum=128,
                                    value=16,
                                    step=4,
                                    info="Higher = more capacity, more VRAM",
                                )
                                train_lora_alpha = gr.Slider(
                                    label="LoRA Alpha",
                                    minimum=4,
                                    maximum=128,
                                    value=16,
                                    step=4,
                                    info="Usually same as rank",
                                )

                        with gr.Column(scale=1):
                            gr.Markdown("#### 📊 Training Parameters")
                            with gr.Row():
                                train_epochs = gr.Slider(
                                    label="Epochs",
                                    minimum=1,
                                    maximum=100,
                                    value=10,
                                    step=1,
                                )
                                train_max_steps = gr.Number(
                                    label="Max Steps (0=use epochs)",
                                    value=0,
                                    info="Override epochs with fixed step count",
                                )

                            train_lr = gr.Number(
                                label="Learning Rate",
                                value=1e-4,
                                info="Recommended: 1e-4 to 5e-5",
                            )

                            with gr.Row():
                                train_batch_size = gr.Slider(
                                    label="Batch Size",
                                    minimum=1,
                                    maximum=8,
                                    value=1,
                                    step=1,
                                )
                                train_grad_accum = gr.Slider(
                                    label="Gradient Accumulation",
                                    minimum=1,
                                    maximum=32,
                                    value=4,
                                    step=1,
                                    info="Effective batch = batch_size × grad_accum",
                                )

                            train_resolution = gr.Dropdown(
                                label="Training Resolution",
                                choices=[256, 512, 768, 1024],
                                value=512,
                                info="Image resolution for training",
                            )

                            train_save_steps = gr.Slider(
                                label="Save Checkpoint Every N Steps",
                                minimum=100,
                                maximum=5000,
                                value=500,
                                step=100,
                            )

                            gr.Markdown("#### ⚡ Optimizer & Scheduler")
                            train_use_8bit_adam = gr.Checkbox(
                                label="Use 8-bit Adam (saves VRAM)",
                                value=True,
                            )
                            train_lr_scheduler = gr.Dropdown(
                                label="LR Scheduler",
                                choices=[
                                    "constant",
                                    "linear",
                                    "cosine",
                                    "cosine_with_restarts",
                                    "polynomial",
                                ],
                                value="cosine",
                            )
                            train_warmup_steps = gr.Slider(
                                label="Warmup Steps",
                                minimum=0,
                                maximum=1000,
                                value=100,
                                step=10,
                            )

                            with gr.Accordion(
                                "🔬 Advanced Training Options", open=False
                            ):
                                train_snr_gamma = gr.Number(
                                    label="SNR Gamma (0=disabled)",
                                    value=5.0,
                                    info="Min-SNR weighting for better convergence",
                                )
                                train_noise_offset = gr.Number(
                                    label="Noise Offset (0=disabled)",
                                    value=0.0,
                                    info="Helps with dark/bright images",
                                )
                                train_seed = gr.Number(
                                    label="Random Seed",
                                    value=42,
                                )

                    gr.Markdown("---")

                    with gr.Row():
                        start_train_btn = gr.Button(
                            "🚀 Start Training",
                            variant="primary",
                            size="lg",
                        )
                        stop_train_btn = gr.Button(
                            "⛔ Stop Training",
                            variant="stop",
                            size="lg",
                        )
                        refresh_log_btn = gr.Button(
                            "🔄 Refresh Log",
                            size="lg",
                        )

                    training_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="⚪ No training started",
                    )

                    training_output = gr.Textbox(
                        label="📋 Training Log",
                        lines=15,
                        max_lines=30,
                        interactive=False,
                    )

                    # Training event handlers
                    start_train_btn.click(
                        fn=self.start_training,
                        inputs=[
                            train_dataset_path,
                            train_output_dir,
                            train_model_id,
                            train_use_tiny,
                            train_lora_rank,
                            train_lora_alpha,
                            train_epochs,
                            train_max_steps,
                            train_lr,
                            train_batch_size,
                            train_grad_accum,
                            train_resolution,
                            train_save_steps,
                            train_use_8bit_adam,
                            train_lr_scheduler,
                            train_warmup_steps,
                            train_snr_gamma,
                            train_noise_offset,
                            train_seed,
                        ],
                        outputs=[training_output],
                    ).then(
                        fn=self.get_training_status,
                        outputs=[training_status],
                    )

                    stop_train_btn.click(
                        fn=self.stop_training,
                        outputs=[training_output],
                    ).then(
                        fn=self.get_training_status,
                        outputs=[training_status],
                    )

                    refresh_log_btn.click(
                        fn=self.get_training_log,
                        outputs=[training_output],
                    ).then(
                        fn=self.get_training_status,
                        outputs=[training_status],
                    )

        demo.launch(server_name=server_name, share=share)


class TrainingOnlyInterface:
    """
    Lightweight interface for training-only mode (no model loading required).
    """

    def __init__(self):
        self.interface = FluxWebInterface(generator=None)

    def launch(self, share: bool = False, server_name: str = "0.0.0.0"):
        self.interface.launch(share=share, server_name=server_name)


# Backwards compatibility alias
WebInterface = FluxWebInterface
