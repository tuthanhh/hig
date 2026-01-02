"""
Gradio Web Interface for Flux.1 Vietnamese Image Generator

Provides an interactive UI for:
- Vietnamese text input with automatic translation
- Image generation with adjustable parameters
- Multiple resolution options optimized for Flux
"""

import gradio as gr
from hig.inference.generator import FluxImageGenerator


class FluxWebInterface:
    """
    Gradio-based web interface for Vietnamese text-to-image generation.
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

    def __init__(self, generator: FluxImageGenerator):
        """
        Args:
            generator: Initialized FluxImageGenerator instance
        """
        self.generator = generator

    def predict(
        self,
        prompt: str,
        resolution: str,
        steps: int,
        guidance: float,
        seed: int,
    ):
        """
        Generate image from Vietnamese prompt.

        Args:
            prompt: Vietnamese text prompt
            resolution: Resolution preset name
            steps: Number of inference steps
            guidance: Guidance scale
            seed: Random seed

        Returns:
            Tuple of (generated image, translated prompt)
        """
        width, height = self.RESOLUTION_PRESETS[resolution]

        image, translated_text = self.generator.generate(
            prompt_vn=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=int(seed),
        )
        return image, translated_text

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
                #  Vietnamese Historical Image Generator
                ### Powered by Flux.1 + Custom LoRA

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

                    resolution = gr.Dropdown(
                        label=" Resolution",
                        choices=list(self.RESOLUTION_PRESETS.keys()),
                        value="Square (1024x1024)",
                    )

                    with gr.Accordion("️ Advanced Settings", open=False):
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
                inputs=[prompt, resolution, steps, guidance, seed],
                outputs=[output_image, translated_output],
            )

            # Example prompts
            gr.Examples(
                examples=[
                    ["Vua Lê Đại Hành trong bộ áo long bào, đứng trước quân đội"],
                    ["Một trận thủy chiến trên sông Bạch Đằng với cọc gỗ"],
                    ["Cảnh chợ quê Việt Nam thời xưa với người bán hàng"],
                    ["Kinh thành Thăng Long với cung điện và thành lũy"],
                ],
                inputs=prompt,
            )

        demo.launch(server_name=server_name, share=share)


# Backwards compatibility alias
WebInterface = FluxWebInterface
