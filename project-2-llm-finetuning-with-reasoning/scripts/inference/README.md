# Inference Scripts

This directory contains prediction and inference scripts for generating competition submissions.

## Scripts Overview

### `predict_test_reasoning_data_qwen2.5.py`
- **Model**: Qwen2.5-7B with reasoning chains
- **Performance**: Best competition results
- **Output**: Structured predictions with reasoning traces
- **Features**:
  - XML reasoning chain generation
  - Batch processing for efficiency
  - Confidence scoring
  - Neutrality validation

### `predict_test_reasoning_data_deepseek.py`
- **Model**: DeepSeek-R1-14B
- **Performance**: Alternative approach validation
- **Output**: Comparative predictions
- **Features**:
  - DeepSeek-specific reasoning format
  - Memory-optimized inference
  - Batch processing support

## Usage

### Basic Inference
```bash
# Generate predictions with Qwen2.5 reasoning model
python scripts/inference/predict_test_reasoning_data_qwen2.5.py

# Generate predictions with DeepSeek model
python scripts/inference/predict_test_reasoning_data_deepseek.py
```

### Advanced Configuration
```bash
python scripts/inference/predict_test_reasoning_data_qwen2.5.py \
    --model_path "models/qwen_reasoning/checkpoint-best" \
    --test_file "data/test_data.csv" \
    --output_file "submission/prediction_qwen.csv" \
    --batch_size 8 \
    --max_length 2048 \
    --temperature 0.1 \
    --top_p 0.9
```

## Output Format

### Standard Predictions
```csv
id,answer
1,A
2,B
3,C
...
```

### Detailed Predictions (with reasoning)
```json
{
  "id": 1,
  "question": "Question text",
  "reasoning": "<question>...</question><think>...</think><reasoning>...</reasoning>...",
  "answer": "A",
  "confidence": 0.95,
  "neutrality_score": 0.87
}
```

## Features

### Reasoning Chain Generation
- **Structured XML output** following training format
- **Multi-step reasoning** with explicit thinking process
- **Reflection and adjustment** phases for accuracy
- **Neutrality enforcement** for sensitive topics

### Performance Optimization
- **Batch processing** for efficient inference
- **Memory management** for large datasets
- **GPU acceleration** with CUDA support
- **Parallel processing** for multiple models

### Quality Assurance
- **Confidence scoring** for prediction reliability
- **Neutrality validation** for political topics
- **Format validation** for submission compliance
- **Error handling** for robust execution

## Model Loading

Scripts automatically handle:
- **Model initialization** from checkpoints
- **Tokenizer configuration** for proper encoding
- **Device allocation** for GPU/CPU usage
- **Memory optimization** for large models

## Submission Generation

The inference scripts generate competition-ready submission files:
- **CSV format** for Kaggle submission
- **ID-answer mapping** as required
- **Validation checks** for format compliance
- **Backup generation** for redundancy

## Performance Metrics

Key metrics tracked during inference:
- **Inference speed**: Questions per second
- **Memory usage**: Peak GPU/CPU utilization
- **Accuracy**: Validation set performance
- **Neutrality**: Bias detection scores