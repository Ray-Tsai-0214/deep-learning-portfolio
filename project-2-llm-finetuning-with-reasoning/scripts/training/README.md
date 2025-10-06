# Training Scripts

This directory contains model training scripts for different architectures and approaches.

## Scripts Overview

### `qwen_finetune_reasoning.py`
- **Model**: Qwen2.5-7B
- **Approach**: Fine-tuning with structured reasoning chains
- **Performance**: Best overall results (Private: 0.72043)
- **Features**:
  - XML-formatted reasoning chain training
  - LoRA/QLoRA parameter-efficient fine-tuning
  - Gradient checkpointing for memory efficiency
  - Multi-step reasoning process learning

### `qwen_finetune_without_reasoning.py`
- **Model**: Qwen2.5-7B/14B
- **Approach**: Standard fine-tuning without reasoning chains
- **Purpose**: Baseline comparison
- **Features**:
  - Direct question-answer training
  - Standard supervised fine-tuning
  - Memory-efficient implementation

### `deepseek_finetune_reasoning.py`
- **Model**: DeepSeek-R1-14B
- **Approach**: Mixed reasoning approach
- **Purpose**: Experimental validation
- **Features**:
  - DeepSeek model architecture
  - Custom reasoning chain adaptation
  - Comparative analysis framework

## Usage

### Basic Training
```bash
# Train Qwen2.5 with reasoning (recommended)
python scripts/training/qwen_finetune_reasoning.py

# Train baseline without reasoning
python scripts/training/qwen_finetune_without_reasoning.py

# Train DeepSeek variant
python scripts/training/deepseek_finetune_reasoning.py
```

### Configuration

Each script supports command-line arguments for customization:

```bash
python scripts/training/qwen_finetune_reasoning.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_path "data/sample_train_data_reasoning.jsonl" \
    --output_dir "models/qwen_reasoning" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --use_lora True \
    --lora_rank 64
```

## Key Features

### Memory Optimization
- **4-bit quantization** for reduced memory usage
- **Gradient checkpointing** for large model training
- **LoRA adapters** for parameter-efficient fine-tuning

### Training Stability
- **Learning rate scheduling** for stable convergence
- **Gradient clipping** to prevent instability
- **Mixed precision training** for speed and memory

### Reasoning Chain Training
- **XML parsing** for structured reasoning
- **Multi-step loss calculation** for chain components
- **Neutrality enforcement** through specialized loss functions

## Performance Monitoring

All scripts integrate with Weights & Biases for comprehensive tracking:
- Training loss progression
- Validation accuracy metrics
- Memory usage monitoring
- GPU utilization tracking

## Hardware Requirements

- **GPU**: RTX 4090 or equivalent (24GB+ VRAM)
- **Memory**: 32GB+ system RAM
- **Storage**: 50GB+ for model checkpoints
- **CUDA**: 11.8+ for optimal performance