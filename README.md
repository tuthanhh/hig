# HIG - Vietnamese Historical Image Generator

A Flux.1-based text-to-image generation system specialized for Vietnamese historical content.

## 🏗️ Architecture

Built on **Flux.1** (Black Forest Labs), featuring:

| Component | Model | Description |
|-----------|-------|-------------|
| **Transformer** | `FluxTransformer2DModel` | MMDiT architecture for denoising |
| **Scheduler** | `FlowMatchEulerDiscreteScheduler` | Rectified Flow for efficient sampling |
| **VAE** | `AutoencoderKL` | Image encoding/decoding |
| **Text Encoder 1** | `CLIPTextModel` | `clip-vit-large-patch14` for pooled embeddings |
| **Text Encoder 2** | `T5EncoderModel` | `google/t5-v1_1-xxl` for sequence embeddings |
| **Tokenizer 1** | `CLIPTokenizer` | CLIP tokenization |
| **Tokenizer 2** | `T5TokenizerFast` | T5 tokenization |

## 📦 Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/tuthanhh/hig.git
cd hig

# Install dependencies (PyTorch will auto-detect CUDA version)
uv sync
```

### CUDA-Specific Installation

The project supports both CUDA 12.x and CUDA 13.x. PyTorch will automatically detect your CUDA version during installation.

**For CUDA 12.x (12.1, 12.4, 12.6):**
```bash
uv sync --extra cuda12
```

**For CUDA 13.x:**
```bash
uv sync --extra cuda13
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

## 🚀 Quick Start

### 1. Preprocess Data

```bash
uv run python src/hig/data/preprocessor.py
```

### 2. Train LoRA

**Production Mode:**
```bash
uv run python -m hig.train \
    --dataset_path data/processed \
    --output_dir output/lora \
    --epochs 1 \
    --lora_rank 16
```

**Tiny Debug Mode (No Model Download, <2GB VRAM):**
```bash
# Uses randomly initialized tiny models for quick debugging
uv run python -m hig.train \
    --dataset_path data/processed \
    --output_dir output/lora_debug \
    --tiny \
    --max_steps 10 \
    --resolution 256
```

### 3. Run Inference

```bash
uv run python -m hig.run_inference \
    --lora_path output/lora \
    --share
```

## 📁 Project Structure

```
hig/
├── scripts/
│   ├── train.py            # Training script
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
│   ├── raw/                # Raw input data
│   └── processed/          # Processed HuggingFace dataset
└── output/
    └── lora/               # Trained LoRA weights
```

## 🔧 Configuration

### Tiny Model Debug Mode

For rapid debugging without downloading large models, use tiny mode with randomly initialized models (<2GB VRAM):

```bash
python -m hig.train --tiny --max_steps 10 --resolution 256
```

**Tiny mode specifications:**
- **CLIP**: 32 hidden_size, 2 layers (randomly initialized)
- **T5**: 64 d_model, 2 layers (randomly initialized)
- **VAE**: madebyollin/taesd (real tiny VAE, ~5MB)
- **Flux Transformer**: 2 layers, 16 attention_head_dim (randomly initialized)
- **Total VRAM**: <2GB
- **Use case**: Testing training pipeline, debugging data loading, quick iterations

⚠️ **Important**: Tiny models are NOT pretrained and will NOT generate meaningful images. Use only for debugging the training code.

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_path` | `data/processed` | Path to processed dataset |
| `--output_dir` | `output/lora` | Output directory for LoRA |
| `--model_id` | `black-forest-labs/FLUX.1-dev` | Base model (ignored if --tiny) |
| `--tiny` | `False` | Use tiny random models for debugging |
| `--max_steps` | `None` | Limit training steps (useful with --tiny) |
| `--lora_rank` | `16` | LoRA rank (higher = more capacity) |
| `--lora_alpha` | `16` | LoRA alpha (scaling factor) |
| `--epochs` | `1` | Training epochs |
| `--lr` | `1e-4` | Learning rate |
| `--batch_size` | `1` | Batch size |
| `--resolution` | `1024` | Training resolution |
| `--no_4bit` | `False` | Disable 4-bit quantization |

## 💡 Usage Examples

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

```python
from hig import FluxModelLoader, FluxLoraAdapter, FluxTrainer
from accelerate import Accelerator

# Load components
loader = FluxModelLoader()
components = loader.load_training_components(load_transformer_in_4bit=True)

# Apply LoRA
adapter = FluxLoraAdapter(rank=16, alpha=16)
components["transformer"] = adapter.apply(components["transformer"])

# Train
trainer = FluxTrainer(components, dataloader, Accelerator(), args)
trainer.train()
trainer.save_lora("output/lora")
```

## 📋 Requirements

- Python 3.13+
- CUDA-capable GPU with 24GB+ VRAM (for 4-bit training)
- HuggingFace account with access to `black-forest-labs/FLUX.1-dev`

## 📄 License

MIT License