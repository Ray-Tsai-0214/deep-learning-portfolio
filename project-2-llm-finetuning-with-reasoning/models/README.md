# Models Directory

This directory contains model configurations, metadata, and related files for the Kaggle #2 reasoning LLM project.

## Files

### `model_config.json`
- **Type**: Configuration file
- **Content**: Model hyperparameters and training settings
- **Usage**: Referenced by training and inference scripts

## Model Architecture

### Qwen2.5-7B with Reasoning Chains
- **Base Model**: Qwen/Qwen2.5-7B-Instruct
- **Fine-tuning**: LoRA/QLoRA parameter-efficient training
- **Specialization**: Structured reasoning chain generation
- **Performance**: Best competition results (Private: 0.72043)

### Model Specifications
```json
{
  "model_name": "Qwen/Qwen2.5-7B-Instruct",
  "architecture": "transformer",
  "parameters": "7B",
  "context_length": 32768,
  "vocabulary_size": 151936,
  "fine_tuning_method": "LoRA",
  "quantization": "4-bit"
}
```

## HuggingFace Model

### Published Model
**[RayTsai/chinese-reasoning-qwen2.5-7b](https://huggingface.co/RayTsai/chinese-reasoning-qwen2.5-7b)**

### Model Card Features
- **Language**: Chinese (Simplified)
- **Task**: Multiple Choice Reasoning
- **Domain**: Political and Sensitive Topics
- **Approach**: Structured Reasoning Chains
- **License**: Apache 2.0

### Usage Example
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("RayTsai/chinese-reasoning-qwen2.5-7b")
model = AutoModelForCausalLM.from_pretrained(
    "RayTsai/chinese-reasoning-qwen2.5-7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate reasoning chain
prompt = "請分析以下政治議題..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=2048)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Configuration

### Fine-tuning Settings
```json
{
  "training_config": {
    "learning_rate": 2e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0
  },
  "lora_config": {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
  },
  "quantization_config": {
    "load_in_4bit": true,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": true,
    "bnb_4bit_quant_type": "nf4"
  }
}
```

### Hardware Requirements
- **GPU**: RTX 4090 (24GB VRAM) or equivalent
- **Memory**: 32GB+ system RAM
- **Storage**: 50GB+ for model and checkpoints
- **CUDA**: 11.8+ for optimal performance

## Model Performance

### Competition Results
| Metric | Score | Rank | Analysis |
|--------|-------|------|----------|
| Private Score | 0.72043 | #30 | Excellent generalization |
| Public Score | 0.47177 | - | Conservative public performance |
| Gap | +0.25 | - | Strong reasoning capability |

### Key Strengths
1. **Reasoning Chain Generation**: Structured XML output format
2. **Neutrality**: Balanced handling of sensitive topics  
3. **Generalization**: Strong performance on unseen data
4. **Consistency**: Reliable reasoning process

### Model Capabilities
- **Multi-step Reasoning**: Complex logical deduction
- **Context Understanding**: Deep comprehension of Chinese political topics
- **Neutrality Maintenance**: Unbiased perspective on sensitive issues
- **Structured Output**: Consistent XML formatting

## Model Variants

### Qwen2.5-7B (Reasoning)
- **Best Performance**: Competition winner
- **Approach**: Structured reasoning chains
- **Training Data**: XML-formatted reasoning examples

### Qwen2.5-14B (Baseline)
- **Purpose**: Comparison baseline
- **Approach**: Standard fine-tuning
- **Training Data**: Direct question-answer pairs

### DeepSeek-R1-14B (Experimental)
- **Purpose**: Alternative architecture validation
- **Approach**: Mixed reasoning methodology
- **Training Data**: Adapted reasoning format

## Deployment

### Local Inference
```bash
# Load model for local inference
python scripts/inference/predict_test_reasoning_data_qwen2.5.py \
    --model_path "RayTsai/chinese-reasoning-qwen2.5-7b" \
    --test_file "data/test.csv" \
    --output_file "predictions.csv"
```

### API Integration
```python
# Integrate with existing API
from transformers import pipeline

reasoning_pipeline = pipeline(
    "text-generation",
    model="RayTsai/chinese-reasoning-qwen2.5-7b",
    tokenizer="RayTsai/chinese-reasoning-qwen2.5-7b",
    torch_dtype=torch.float16
)

result = reasoning_pipeline(prompt, max_length=2048)
```

## Model Maintenance

### Monitoring Metrics
- **Inference Speed**: Tokens per second
- **Memory Usage**: GPU/CPU utilization
- **Quality Scores**: Reasoning chain validation
- **Neutrality Scores**: Bias detection results

### Update Procedures
- **Version Control**: Track model iterations
- **Performance Testing**: Validate improvements
- **Documentation**: Update model cards
- **Distribution**: Sync with HuggingFace Hub

This model represents a significant advancement in Chinese reasoning LLM capabilities, particularly for sensitive political topics requiring neutral, structured analysis.