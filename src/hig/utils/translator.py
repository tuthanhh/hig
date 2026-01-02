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
            print("Translating single string...")
            print(f"Input VN: {text}")
            res = self._generate_translation(self._clean_content(text))
            print(f"Output EN: {res}")
            return res
        elif isinstance(text, list):
            # GGUF doesn't support true batching like PyTorch, so we loop.
            # This is still fast enough for preprocessing.
            print(f"Translating list of {len(text)} strings...")
            results = []
            for idx, item in enumerate(text):
                print(f"\n[{idx + 1}/{len(text)}] Input VN: {item}")
                cleaned = self._clean_content(item)
                translated = self._generate_translation(cleaned)
                print(f"[{idx + 1}/{len(text)}] Output EN: {translated}")
                results.append(translated)
            return results
        else:
            raise ValueError("Input text must be a string or a list of strings.")

    import re

    def _clean_content(self, text: str) -> str:
        """
        Converts structured Vietnamese image descriptions into a clean,
        paragraph-format Vietnamese description.

        Removes headers, markdown, numbering, and meta-instructions (Lưu ý)
        without translating.
        """
        system_content = (
            "You are a professional Vietnamese text editor. "
            "Your task is to rewrite structured image descriptions into a single, fluent Vietnamese paragraph. "
            "STRICT RULES:\n"
            "1. Do NOT translate. Keep the output in Vietnamese.\n"
            "2. Remove all section headers (e.g., '1. Bối cảnh', '2. Mô tả'), bullet points, and numbering.\n"
            "3. Remove Markdown formatting (such as **bold** or *italics*).\n"
            "4. Remove meta-instructions or notes (specifically the 'Lưu ý' section).\n"
            "5. Merge the remaining sentences into a natural flow."
        )

        # One-shot example from the dataset
        example_input = """Dưới đây là mô tả chi tiết hình ảnh:

**1. Bối cảnh, đối tượng chính, màu sắc và hành động:**

Hình ảnh là một biểu tượng tròn, có vẻ như là một ấn phẩm sách hoặc tài liệu. Phần nền của biểu tượng là một họa tiết hình tròn phức tạp, có các hình ảnh động vật và con người được khắc tỉ mỉ, tạo cảm giác như một bản đồ hoặc một hình ảnh mang tính biểu tượng. Bên trong hình tròn là một vòng tròn trắng, làm nổi bật các yếu tố chính.

Đối tượng chính là dòng chữ màu đen lớn, in trên vòng tròn trắng.

Màu sắc chủ đạo là đen, trắng và xám. Màu đen được sử dụng cho chữ và các chi tiết trên họa tiết nền, tạo sự tương phản mạnh mẽ với màu trắng.

Hành động không rõ ràng, hình ảnh chủ yếu là tĩnh, thể hiện một ấn phẩm hoặc tài liệu.

**2. Mô tả con người/nhân vật (nếu có):**

Hình ảnh không có con người hoặc nhân vật rõ ràng.

**3. Trích dẫn nội dung chữ:**

Dòng chữ chính là: **"LỊCH SỬ VIỆT NAM BẢNG TRANH"** (Lịch sử Việt Nam tranh)

Dưới dòng chữ chính là: **"TẬP 12: CỔ LAU VẠN THÁNG VƯƠNG"** (Tập 12: Cổ Lau Vạn Tháng Vương)

**4. Lưu ý:**

Không giả định thể loại hoặc ngữ cảnh nếu không được thể hiện rõ.

Không dùng các câu dẫn nhập như 'Hình ảnh cho thấy', 'Bức tranh mô tả'."""

        # Updated output: Vietnamese, paragraph form, no headers, no "Lưu ý"
        example_output = (
            "Hình ảnh là một biểu tượng tròn, có vẻ như là một ấn phẩm sách hoặc tài liệu. "
            "Phần nền của biểu tượng là một họa tiết hình tròn phức tạp, có các hình ảnh động vật và con người được khắc tỉ mỉ, "
            "tạo cảm giác như một bản đồ hoặc một hình ảnh mang tính biểu tượng. "
            "Bên trong hình tròn là một vòng tròn trắng, làm nổi bật các yếu tố chính. "
            "Đối tượng chính là dòng chữ màu đen lớn, in trên vòng tròn trắng. "
            "Màu sắc chủ đạo là đen, trắng và xám; màu đen được sử dụng cho chữ và các chi tiết trên họa tiết nền, "
            "tạo sự tương phản mạnh mẽ với màu trắng. Hình ảnh chủ yếu là tĩnh, thể hiện một ấn phẩm hoặc tài liệu. "
            "Hình ảnh không có con người hoặc nhân vật rõ ràng. "
            'Dòng chữ chính là "LỊCH SỬ VIỆT NAM BẢNG TRANH" (Lịch sử Việt Nam tranh), '
            'và dưới đó là "TẬP 12: CỔ LAU VẠN THÁNG VƯƠNG" (Tập 12: Cổ Lau Vạn Tháng Vương).'
        )

        # Flag to suppress reasoning/thinking in some models (if supported by your prompt template)
        processed_input = text + "/no_think"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": example_input},
            {"role": "assistant", "content": example_output},
            {"role": "user", "content": processed_input},
        ]

        try:
            output = self.llm.create_chat_completion(
                messages=messages,
                temperature=0.1,  # Keep low for deterministic cleaning
                max_tokens=1024,
            )
            cleaned_content = output["choices"][0]["message"]["content"]

            # Remove thinking tags if the model outputs them despite instructions
            cleaned_content = re.sub(
                r"<think>.*?</think>", "", cleaned_content, flags=re.DOTALL
            )

            return cleaned_content.strip()

        except Exception as e:
            print(f"Content Cleaning Error: {e}")
            return text  # Fallback to original text if failure

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
