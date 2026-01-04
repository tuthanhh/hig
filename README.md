# HIG - Vietnamese Historical Image Generator

A Flux.1-based text-to-image generation system specialized for Vietnamese historical content.

## Installation

### Prerequisites

- **uv**: Fast Python package installer ([installation guide](https://github.com/astral-sh/uv))
- **CUDA 12.6**: This project targets CUDA 12.6 specifically
- **C++ Compiler**: Required for building `llama-cpp-python` (e.g., Visual Studio Build Tools on Windows, GCC on Linux)
- **CMake**: Required for compiling CUDA support in llama-cpp-python

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/tuthanhh/hig.git
cd hig

# Set environment variable for llama-cpp-python CUDA support
$env:CMAKE_ARGS="-DGGML_CUDA=on"  # PowerShell
# OR
export CMAKE_ARGS="-DGGML_CUDA=on"  # Bash/Linux

# Install dependencies
uv sync
```

## Quick Start

### 1. Prepare Dataset

Preprocess your data by validating images and translating Vietnamese captions to English:

```bash
# Process all JSONL files in a directory
uv run scripts/prepare_data.py \
    --jsonl_dir data/raw/04_output \
    --output_dir data/processed/dataset

# Or process specific JSONL files
uv run scripts/prepare_data.py \
    --jsonl_paths data/raw/04_output/metadata.jsonl \
    --output_dir data/processed/dataset
```

### 2. Export Dataset (Optional)

Export images and text files for manual inspection or external tools:

```bash
# Export from HuggingFace dataset
uv run scripts/export_dataset.py \
    --input data/processed/dataset \
    --output data/exported \
    --caption_lang en

# Export from JSONL file
uv run scripts/export_dataset.py \
    --input data/processed/dataset/sample_translations.jsonl \
    --output data/exported \
    --caption_lang vn
```

### 3. Train LoRA

**Note:** This project uses [ai-toolkit](https://github.com/ostris/ai-toolkit) for training LoRA adapters, not the built-in training scripts.

Training is performed using the Google Colab notebook:
- [HIG Training Notebook](https://colab.research.google.com/drive/1bwk969Nu7i3wf8w_cMJtyJtf0K99NS3A?usp=sharing)

The notebook uses ai-toolkit to fine-tune Flux.1 with your prepared dataset.

### 4. Run Inference

**Basic Usage:**
```bash
uv run scripts/run_inference.py \
    --lora_path output/lora \
    --share
```

**Memory-Optimized Mode:**
```bash
# 4-bit quantization (saves ~18GB VRAM)
uv run scripts/run_inference.py \
    --quantization 4bit \
    --lora_path output/lora

# 8-bit quantization (saves ~12GB VRAM)
uv run scripts/run_inference.py \
    --quantization 8bit \
    --lora_path output/lora
```

**Cloud-Based Inference (No Local Hardware Required):**

If your local hardware is not sufficient, use the Google Colab notebook:
- [HIG Inference Notebook](https://colab.research.google.com/drive/1q5TH4fMKjtPF3N0Wm-gPgJBP1nyxnBOM?usp=sharing)

## Project Structure

```
hig/
├── scripts/
│   ├── prepare_data.py     # Data preprocessing & translation
│   ├── export_dataset.py   # Export dataset to images + text files
│   ├── train.py            # Training script (deprecated - use ai-toolkit)
│   └── run_inference.py    # Inference/web UI script
├── src/hig/
│   ├── data/
│   │   ├── dataset.py      # FluxDataset for training
│   │   └── preprocessor.py # Data preprocessing with translation
│   ├── model/
│   │   ├── loader.py       # FluxModelLoader with tiny mode
│   │   └── adapter.py      # FluxLoraAdapter (PEFT)
│   ├── inference/
│   │   ├── generator.py    # FluxImageGenerator
│   │   └── interface.py    # Gradio web UI
│   ├── utils/
│   │   └── translator.py   # Vietnamese-English translation
│   └── trainer.py          # FluxTrainer with Flow Matching
├── data/
│   ├── raw/                # Raw input data (JSONL files)
│   ├── processed/          # Processed HuggingFace dataset
│   └── exported/           # Exported images and text files
└── output/
    └── lora/               # Trained LoRA weights
```

---

## Scripts Reference

### 1. `prepare_data.py` - Dataset Preprocessing

Validates images and translates Vietnamese captions to English using a GGUF model.

**Basic Usage:**

```bash
# Process all JSONL files in a directory
uv run scripts/prepare_data.py \
    --jsonl_dir data/raw/04_output \
    --output_dir data/processed/dataset

# Process specific JSONL files
uv run scripts/prepare_data.py \
    --jsonl_paths data/raw/file1.jsonl data/raw/file2.jsonl \
    --output_dir data/processed/dataset
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--jsonl_paths` | `str+` | *Required* | Path(s) to JSONL file(s) containing image paths and captions |
| `--jsonl_dir` | `str` | *Required* | Directory containing JSONL files (all .jsonl files processed) |
| `--output_dir` | `str` | `data/processed/dataset` | Where to save the processed HuggingFace dataset |
| `--image_root_override` | `str` | `None` | New root folder for images if different from JSONL |
| `--model_path` | `str` | `None` | Path to local .gguf Qwen model (auto-downloads if not specified) |
| `--n_gpu_layers` | `int` | `-1` | Number of layers to offload to GPU (-1 = all layers) |
| `--include_figures` | flag | `True` | Include figure_detail entries from JSONL files |
| `--no_figures` | flag | - | Exclude figure_detail entries from JSONL files |

**Input Format (JSONL):**

Each line should be a JSON object with:
```json
{
  "image": "data/raw/images/001.jpg",
  "caption_detail": "Một người nông dân đang cày ruộng",
  "figure_detail": "Người nông dân mặc áo bà ba màu nâu"
}
```

**Output:**

Saves a HuggingFace dataset with columns:
- `image`: PIL Image object
- `caption_vn`: Original Vietnamese caption
- `caption_en`: Translated English caption

---

### 2. `export_dataset.py` - Dataset Export

Exports processed dataset as paired image and text files for external tools or manual inspection.

**Basic Usage:**

```bash
# Export from HuggingFace dataset folder
uv run scripts/export_dataset.py \
    --input data/processed/dataset \
    --output data/exported \
    --caption_lang en

# Export from JSONL file
uv run scripts/export_dataset.py \
    --input data/processed/dataset/sample_translations.jsonl \
    --output data/exported \
    --caption_lang vn
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | `str` | *Required* | Path to HuggingFace dataset folder or JSONL file |
| `--output` | `str` | *Required* | Output directory for exported files |
| `--caption_lang` | `str` | `en` | Caption language to export (`en` or `vn`) |
| `--base_path` | `str` | `None` | Base path for resolving relative image paths (JSONL only) |
| `--image_column` | `str` | `image` | Name of image column (HuggingFace dataset only) |
| `--caption_column` | `str` | `None` | Name of caption column (auto-detected if not specified) |

**Output Format:**

Creates paired files with matching names:
```
data/exported/
├── image_0000.png
├── image_0000.txt
├── image_0001.png
├── image_0001.txt
└── ...
```

---

### 3. `train.py` - LoRA Training (Deprecated)

**Status:** This script is deprecated. Training is now performed using [ai-toolkit](https://github.com/ostris/ai-toolkit) via the [HIG Training Notebook](https://colab.research.google.com/drive/1bwk969Nu7i3wf8w_cMJtyJtf0K99NS3A?usp=sharing).

The ai-toolkit provides more robust training capabilities and better results for fine-tuning Flux.1 models with LoRA adapters

---

### 4. `run_inference.py` - Web Interface & Inference

Launches a Gradio web interface for image generation.

**Cloud Alternative:** If local hardware is insufficient, use the [HIG Inference Notebook](https://colab.research.google.com/drive/1q5TH4fMKjtPF3N0Wm-gPgJBP1nyxnBOM?usp=sharing) on Google Colab.

**Note:** Training is performed using [ai-toolkit](https://github.com/ostris/ai-toolkit). This script is for inference only.

**Basic Usage:****

```bash
# Launch inference interface
uv run scripts/run_inference.py \
    --lora_path output/lora \
    --share

# Memory-optimized with 4-bit quantization
uv run scripts/run_inference.py \
    --quantization 4bit \
    --lora_path output/lora

# Memory-optimized with 8-bit quantization
uv run scripts/run_inference.py \
    --quantization 8bit \
    --lora_path output/lora
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_id` | `str` | `black-forest-labs/FLUX.1-dev` | Flux model ID |
| `--lora_path` | `str` | `None` | Path to trained LoRA weights |
| `--translator_path` | `str` | `None` | Path to GGUF translator model (auto-downloads if not specified) |
| `--share` | flag | `False` | Create public Gradio share link |
| `--server_name` | `str` | `0.0.0.0` | Server hostname |
| `--quantization` | `str` | `None` | Load model with quantization (`4bit` or `8bit`) |
| `--no-cpu-offload` | flag | `False` | Disable CPU offloading (use more VRAM but faster) |

**Memory Optimization:**

- **Standard Mode**: Full model in VRAM
  - Requires: 24GB+ VRAM
  - Features: Vietnamese prompt input, real-time image generation

- **Quantized Mode**: Reduced memory footprint
  - 4-bit: ~18GB VRAM savings (requires 12GB+ VRAM)
  - 8-bit: ~12GB VRAM savings (requires 16GB+ VRAM)
  - Requires: `bitsandbytes` library

**Interface Features:**
- Vietnamese prompt support (auto-translation)
- Real-time image generation
- Generated image gallery
- LoRA weight loading

---

## Configuration

### Complete Workflow Example

```bash
# 1. Prepare dataset (validate images + translate captions)
uv run scripts/prepare_data.py \
    --jsonl_dir data/raw/04_output \
    --output_dir data/processed/dataset \
    --n_gpu_layers -1

# 2. (Optional) Export for inspection
uv run scripts/export_dataset.py \
    --input data/processed/dataset \
    --output data/exported \
    --caption_lang en

# 3. Train LoRA adapter using ai-toolkit
# Training is performed via Google Colab:
# https://colab.research.google.com/drive/1bwk969Nu7i3wf8w_cMJtyJtf0K99NS3A?usp=sharing

# 4. Run inference with trained model
uv run scripts/run_inference.py \
    --lora_path output/lora \
    --share
```

### Training with ai-toolkit

LoRA training is performed using [ai-toolkit](https://github.com/ostris/ai-toolkit), which provides:
- Robust Flux.1 fine-tuning capabilities
- Efficient LoRA adapter training
- Better convergence and results

Use the [HIG Training Notebook](https://colab.research.google.com/drive/1bwk969Nu7i3wf8w_cMJtyJtf0K99NS3A?usp=sharing) for training your LoRA adapters.

---

## Usage Examples

### Python API

```python
from hig import FluxImageGenerator

# Initialize generator
generator = FluxImageGenerator(
    lora_weights_path="output/lora"
)

# Generate image from Vietnamese prompt
image, translated = generator.generate(
    prompt_vn="Vua Lê Đại Hành cưỡi ngựa ra trận",
    width=1024,
    height=1024,
    num_inference_steps=28,
)

image.save("output.png")
```

### Training API

**Note:** The `hig.train` module is deprecated. Use the `scripts/train.py` script instead (see [Scripts Reference](#scripts-reference)).

## Requirements

- Python 3.13+
- CUDA 12.6
- CUDA-capable GPU with 24GB+ VRAM (for 4-bit training)
- C++ compiler (Visual Studio Build Tools on Windows, GCC on Linux)
- CMake (for llama-cpp-python CUDA support)
- HuggingFace account with access to `black-forest-labs/FLUX.1-dev`

## License

MIT License