import re
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from typing import Union, List, Optional

"""
Translator module using GGUF models via llama.cpp for Vietnamese to English translation.
This module performs 2 main functions:
1. A class that wraps around a GGUF model to perform translations.
2. self.translate() taking in Vietnamese text/list of texts and returning English translations.
"""


class VNTranslator:
    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_id: str = "Qwen/Qwen3-0.6B-GGUF",
        filename: str = "Qwen3-0.6B-Q8_0.gguf",
        n_gpu_layers: int = -1,  # -1 = Offload all layers to GPU
        n_ctx: int = 4096,  # Context window
        verbose: bool = False,
    ):
        """
        Initializes the Translator using a quantized GGUF model via llama.cpp.

        Args:
            model_path: Local path to .gguf file. If None, downloads from HF.
            repo_id: Hugging Face repository ID (used if model_path is None).
            filename: Specific .gguf filename (used if model_path is None).
            n_gpu_layers: Layers to put on GPU. -1 for all (fastest).
            n_ctx: Context window size.
        """

        # 1. Locate or Download Model automatically
        if model_path is None:
            print(f"Translator: Checking cache for {repo_id}...")
            try:
                model_path = hf_hub_download(repo_id=repo_id, filename=filename)
                print(f"Translator: Model loaded from {model_path}")
            except Exception as e:
                raise RuntimeError(f"Could not download model from HF: {e}")

        self.model_path = model_path

        # 2. Initialize Llama Engine
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                verbose=verbose,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize llama-cpp. Ensure GPU drivers are set up. Error: {e}"
            )

    def __del__(self):
        """Cleanup method to properly close the model."""
        try:
            if hasattr(self, "llm") and self.llm is not None:
                self.llm.close()
        except Exception:
            pass  # Ignore cleanup errors

    def translate(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Translates text from Vietnamese to English.
        """
        if isinstance(text, str):
            return self._generate_translation(text)
        elif isinstance(text, list):
            # GGUF doesn't support true batching like PyTorch, so we loop.
            # This is still fast enough for preprocessing.
            return [self._generate_translation(t) for t in text]
        else:
            raise ValueError("Input text must be a string or a list of strings.")

    def _generate_translation(self, text: str) -> str:
        # Strict system prompt to ensure clean output
        system_content = (
            "You are a professional translator. "
            "Translate the following Vietnamese text to English. "
            "Return ONLY the English translation. Do not include original text or explanations."
        )
        # Remove this line to switch to non-thinking mode
        text = text + "/no_think"
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": text},
        ]

        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.1,  # Low temperature for deterministic results
                max_tokens=1024,
            )
            # Extract only the content
            translation = output["choices"][0]["message"]["content"]

            # Remove thinking tags if present
            translation = re.sub(
                r"<think>.*?</think>", "", translation, flags=re.DOTALL
            )

            return translation.strip()

        except Exception as e:
            print(f"Translation Error: {e}")
            return text  # Fallback to original if failure


# Quick Test Block
if __name__ == "__main__":
    # This will download the model (~5GB) on first run
    translator = VNTranslator()

    vn = "Một bức tranh sơn dầu vẽ cảnh phố cổ Hà Nội vào mùa thu."
    print(f"VN: {vn}")
    print(f"EN: {translator.translate(vn)}")
