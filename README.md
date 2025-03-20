# nanogpt2
Personal Projects | Decoder-only transformer model for language modeling.

---

# NanoGPT2: GPT-2 Implementation from Scratch

A PyTorch implementation of OpenAI's GPT-2 model architecture from scratch.

## Overview

This project is a thorough implementation of OpenAI's GPT-2 language model using PyTorch. The main goal is to replicate the architecture and functionality of GPT-2 while providing a clean, educational codebase.

The key difference between this implementation (nanoGPT2) and the original [nanoGPT](https://github.com/Asura3301/nanogpt) is that GPT2:
- Uses LayerNorm instead of BatchNorm
- Moves the normalization to the input of each sub-block
- Adds an additional layer normalization after the final self-attention block

## Installation

```bash
# Clone the repository
git clone https://github.com/Asura3301/nanogpt2.git
cd nanogpt2

# Install required packages
pip install -r requirements.txt
```

## Model Architecture

The implementation follows the original GPT-2 architecture:

- Decoder-only transformer model
- Multi-head self-attention mechanism with causal masking
- Layer normalization before each sub-block (Pre-LN)
- Positional embeddings added to token embeddings
- Feed-forward networks with GELU activation

Model variants:
- GPT-2 (12 layers, 12 heads, 768 embedding dimension, 124M parameters) -> we reproduce this version
- GPT-2 Medium (24 layers, 16 heads, 1024 embedding dimension, 350M parameters)
- GPT-2 Large (36 layers, 20 heads, 1280 embedding dimension, 774M parameters)
- GPT-2 XL (48 layers, 25 heads, 1600 embedding dimension, 1558M parameters)

## Training

The model is trained on the FineWeb-Edu dataset. The training script includes several optimizations:

```bash
# Download and prepare the FineWeb-Edu dataset
python fineweb_edu.py

# Train the model
python train_nanogpt2.py
```

### Training Optimizations

- **Flash Attention**: Implementation of Flash Attention for up to 10x speedup in attention computation
- **Mixed Precision**: Using bfloat16 for 30% faster training
- **Gradient Accumulation**: Allows training with larger effective batch sizes
- **Learning Rate Scheduling**: Cosine decay learning rate scheduler
- **Weight Decay Regularization**: Improves model generalization
- **Distributed Training**: Support for multi-GPU training with DDP

### Training Parameters

- Token sequence length: 1024
- Total batch size: 524,288 tokens
- Micro batch size: 16
- Learning rate: 6e-4 with cosine decay
- Weight decay: 0.1
- Model parameters adapted to be "nice" numbers (divisible by 2)

## Evaluation

The model is evaluated on the HellaSwag dataset, which tests commonsense reasoning and natural language inference.

```bash
# Evaluate the model on HellaSwag
python hellaswag.py --model_type gpt2 --device cuda
```

## Performance Notes

Performance benchmarks on an NVIDIA 3080 Ti GPU:

- **FP32/TF32**: 1079-1156 tokens/sec (14-22 seconds per batch)
- **BFloat16**: 1447-1578 tokens/sec (10-12 seconds per batch), ~30% faster
- **Flash Attention**: 10,176-12,706 tokens/sec (1.2-1.6 seconds per batch), ~10x faster
- **Optimized Model Config**: 12,646-16,315 tokens/sec (1.0-1.4 seconds per batch), ~15% additional gain
- **With LR Scheduler**: 17,251 tokens/sec
- **With Weight Decay & FusedAdamW**: 17,207 tokens/sec
- **With Gradient Accumulation**: 16,000 tokens/sec (higher batch size impacts speed but improves generalization)
- **With DDP**: Performance scales with multiple GPUs (no direct benefit on single GPU)

## Code Features

- Ability to load weights from Hugging Face's GPT-2 models
- Text generation capabilities
- Distributed training support
- Comprehensive logging and checkpointing
- Evaluation on HellaSwag benchmark
- Platform-aware optimizations (e.g., torch.compile on Linux)

## References

1. [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830)
4. [HuggingFace FineWeb-Edu dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
5. [Andrej Karpathy](https://github.com/karpathy)

## License

MIT

---

*Note: This implementation is for educational purposes and is a reference for understanding the GPT-2 architecture.*
