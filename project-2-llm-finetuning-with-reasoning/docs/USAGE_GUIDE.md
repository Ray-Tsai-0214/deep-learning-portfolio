# Usage Guide: Kaggle #2 Reasoning LLM System

## Table of Contents
1. [Quick Start](#quick-start)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Inference and Prediction](#inference-and-prediction)
6. [N8N Workflow Setup](#n8n-workflow-setup)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Quick Start

### Prerequisites Check
```bash
# Check GPU availability
nvidia-smi

# Check Python version (3.8+ required)
python --version

# Check CUDA version (11.8+ recommended)
nvcc --version
```

### 5-Minute Setup
```bash
# 1. Clone repository
git clone https://github.com/Deep-Learning-NYCU/kaggle-2-reasoning-llm-sft-with-reasoning-information-Ray-Tsai-0214.git
cd kaggle-2-reasoning-llm-sft-with-reasoning-information-Ray-Tsai-0214

# 2. Install dependencies
pip install -r requirements.txt

# 3. Quick test with pre-trained model
python scripts/inference/predict_test_reasoning_data_qwen2.5.py --quick_test

# 4. Generate sample predictions
python scripts/inference/predict_test_reasoning_data_qwen2.5.py \
    --model_path "RayTsai/chinese-reasoning-qwen2.5-7b" \
    --test_file "data/sample_test.csv" \
    --output_file "quick_predictions.csv"
```

## Environment Setup

### Hardware Requirements

#### Minimum Requirements
- **GPU**: RTX 3090 (24GB VRAM)
- **CPU**: 8+ cores
- **RAM**: 32GB system memory
- **Storage**: 50GB free space

#### Recommended Requirements
- **GPU**: RTX 4090 (24GB VRAM)
- **CPU**: 12+ cores (AMD Ryzen 9 or Intel i9)
- **RAM**: 64GB system memory
- **Storage**: 100GB NVMe SSD

### Software Dependencies

#### Core Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and related packages
pip install transformers>=4.36.0
pip install datasets>=2.14.0
pip install accelerate>=0.20.0
pip install peft>=0.6.0

# Install additional utilities
pip install pandas numpy scikit-learn
pip install tqdm wandb
pip install bitsandbytes  # For quantization
```

#### Optional Dependencies
```bash
# For development and debugging
pip install jupyter notebook ipython
pip install matplotlib seaborn plotly

# For advanced monitoring
pip install tensorboard
pip install gpustat
```

### Environment Configuration

#### CUDA Environment
```bash
# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6"  # For RTX 4090
export CUDA_LAUNCH_BLOCKING=1  # For debugging
```

#### Python Environment
```bash
# Create conda environment (recommended)
conda create -n kaggle2-reasoning python=3.9
conda activate kaggle2-reasoning

# Or use virtual environment
python -m venv kaggle2-env
source kaggle2-env/bin/activate  # Linux/Mac
# kaggle2-env\Scripts\activate  # Windows
```

## Data Preparation

### Training Data Format

#### Standard JSONL Format
```json
{
  "id": 1,
  "question": "中國的經濟發展模式對環境保護有何影響？",
  "options": [
    "A) 完全負面影響",
    "B) 整體正面但有局部挑戰",
    "C) 影響中性",
    "D) 完全正面影響"
  ],
  "answer": "B",
  "topic": "環境保護",
  "difficulty": "中級"
}
```

#### Reasoning Chain Format
```json
{
  "id": 1,
  "question": "中國的經濟發展模式對環境保護有何影響？",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "reasoning": "<question>中國的經濟發展模式對環境保護有何影響？</question><think>需要客觀分析中國經濟發展與環境保護的關係</think><reasoning>step 1: 分析中國經濟發展的特點... step 2: 評估環境保護政策的實施...</reasoning><reflection>確保分析客觀中立</reflection><adjustment>調整表述確保平衡</adjustment><output>B</output>",
  "answer": "B",
  "topic": "環境保護",
  "difficulty": "中級"
}
```

### Data Validation

#### Format Validation Script
```python
# scripts/utils/validate_data.py
import json
import pandas as pd
from typing import List, Dict

def validate_jsonl_format(file_path: str) -> bool:
    """Validate JSONL file format"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                data = json.loads(line.strip())
                # Check required fields
                required_fields = ['question', 'options', 'answer']
                for field in required_fields:
                    if field not in data:
                        print(f"Line {line_no}: Missing field '{field}'")
                        return False
                        
                # Validate answer format
                if data['answer'] not in ['A', 'B', 'C', 'D']:
                    print(f"Line {line_no}: Invalid answer '{data['answer']}'")
                    return False
                    
        print("✓ Data format validation passed")
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

# Usage
validate_jsonl_format("data/training_data.jsonl")
```

#### Reasoning Chain Validation
```python
import xml.etree.ElementTree as ET

def validate_reasoning_chain(reasoning_text: str) -> bool:
    """Validate XML reasoning chain structure"""
    required_tags = ['question', 'think', 'reasoning', 'reflection', 'adjustment', 'output']
    
    try:
        # Wrap in root element for parsing
        wrapped_xml = f"<root>{reasoning_text}</root>"
        root = ET.fromstring(wrapped_xml)
        
        # Check all required tags present
        found_tags = [elem.tag for elem in root]
        missing_tags = set(required_tags) - set(found_tags)
        
        if missing_tags:
            print(f"Missing tags: {missing_tags}")
            return False
            
        # Validate output format
        output_elem = root.find('output')
        if output_elem is not None and output_elem.text not in ['A', 'B', 'C', 'D']:
            print(f"Invalid output: {output_elem.text}")
            return False
            
        return True
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return False
```

## Model Training

### Basic Training Command
```bash
# Train Qwen2.5 with reasoning chains
python scripts/training/qwen_finetune_reasoning.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_path "data/sample_train_data_reasoning.jsonl" \
    --output_dir "models/qwen_reasoning" \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-5
```

### Advanced Training Configuration
```bash
python scripts/training/qwen_finetune_reasoning.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_path "data/sample_train_data_reasoning.jsonl" \
    --output_dir "models/qwen_reasoning_advanced" \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --save_steps 500 \
    --eval_steps 250 \
    --logging_steps 50 \
    --use_lora true \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.1 \
    --load_in_4bit true \
    --use_gradient_checkpointing true \
    --dataloader_num_workers 0 \
    --seed 42
```

### Training Monitoring

#### Using Weights & Biases
```python
# In training script
import wandb

# Initialize wandb
wandb.init(
    project="kaggle2-reasoning-llm",
    name="qwen2.5-7b-reasoning",
    config={
        "model": "Qwen2.5-7B",
        "method": "LoRA",
        "learning_rate": 2e-5,
        "batch_size": 4
    }
)

# Log metrics during training
wandb.log({
    "train_loss": loss.item(),
    "learning_rate": scheduler.get_last_lr()[0],
    "epoch": epoch
})
```

#### Manual Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f models/qwen_reasoning/training.log

# Check checkpoint directory
ls -la models/qwen_reasoning/checkpoint-*/
```

### Training Troubleshooting

#### Common Issues and Solutions

**CUDA Out of Memory**
```bash
# Solution 1: Reduce batch size
--batch_size 1 --gradient_accumulation_steps 16

# Solution 2: Enable gradient checkpointing
--use_gradient_checkpointing true

# Solution 3: Use 4-bit quantization
--load_in_4bit true
```

**Slow Training Speed**
```bash
# Solution 1: Disable workers for pickle compatibility
--dataloader_num_workers 0

# Solution 2: Use mixed precision
--fp16 true

# Solution 3: Increase batch size if memory allows
--batch_size 8 --gradient_accumulation_steps 2
```

## Inference and Prediction

### Basic Inference
```bash
# Generate predictions with trained model
python scripts/inference/predict_test_reasoning_data_qwen2.5.py \
    --model_path "models/qwen_reasoning/checkpoint-best" \
    --test_file "data/test.csv" \
    --output_file "submission.csv"
```

### Advanced Inference Configuration
```bash
python scripts/inference/predict_test_reasoning_data_qwen2.5.py \
    --model_path "RayTsai/chinese-reasoning-qwen2.5-7b" \
    --test_file "data/test.csv" \
    --output_file "submission_detailed.csv" \
    --batch_size 8 \
    --max_length 2048 \
    --temperature 0.1 \
    --top_p 0.9 \
    --do_sample false \
    --num_beams 1 \
    --save_reasoning true \
    --validate_output true
```

### Batch Processing
```python
# scripts/inference/batch_inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm

def batch_inference(model_path: str, test_file: str, batch_size: int = 8):
    """Process multiple questions in batches"""
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load test data
    df = pd.read_csv(test_file)
    
    results = []
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        
        # Prepare batch inputs
        prompts = [format_prompt(row) for _, row in batch.iterrows()]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)
        
        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=2048,
                temperature=0.1,
                do_sample=False
            )
        
        # Process outputs
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            answer = extract_answer(response)
            results.append({
                'id': batch.iloc[j]['id'],
                'answer': answer,
                'reasoning': response
            })
    
    return results
```

## N8N Workflow Setup

### Installation Options

#### Option 1: Self-Hosted N8N
```bash
# Install Node.js (required)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install N8N globally
npm install n8n -g

# Start N8N
n8n start

# Access at http://localhost:5678
```

#### Option 2: N8N Cloud
1. Visit [n8n.cloud](https://n8n.cloud)
2. Create account and workspace
3. Import workflow JSON file

### Workflow Import Process

1. **Download Workflow File**
   ```bash
   # Workflow file location
   n8n_workflow/DL_data_prepare_muti_topics_gen_repeat.json
   ```

2. **Import in N8N Interface**
   - Open N8N web interface
   - Go to Settings → Import from File
   - Select the JSON file
   - Click Import

3. **Configure Credentials**
   ```json
   {
     "openai_api_key": "sk-your-openai-key-here",
     "api_organization": "your-org-id",
     "model": "gpt-4",
     "temperature": 0.7
   }
   ```

### Workflow Configuration

#### Key Parameters
```json
{
  "topics": [
    "科技倫理", "經濟發展", "環境保護", "教育體系", "網絡安全",
    "法律制度", "人權議題", "政治治理", "言論自由", "宗教文化",
    "民族關係", "地區自治", "國際關係", "社會問題", "歷史事件"
  ],
  "difficulty_levels": ["基礎", "中級", "高級"],
  "question_types": ["事實性", "概念性", "分析性", "比較性", "評估性"],
  "iterations_per_config": 3,
  "output_format": "jsonl"
}
```

#### Output Configuration
```json
{
  "output_path": "/path/to/generated_data/",
  "file_naming": "reasoning_data_{timestamp}.jsonl",
  "batch_size": 25,
  "quality_threshold": 0.8
}
```

### Workflow Monitoring

#### Progress Tracking
```javascript
// N8N function node for progress tracking
const currentProgress = $node["Loop Over Items"].item.index;
const totalItems = $node["Loop Over Items"].item.length;
const progressPercent = (currentProgress / totalItems) * 100;

console.log(`Progress: ${progressPercent.toFixed(1)}%`);

return {
  progress: progressPercent,
  current: currentProgress,
  total: totalItems,
  timestamp: new Date().toISOString()
};
```

#### Quality Monitoring
```javascript
// Quality validation node
const generatedText = $input.first().json.generated_text;
const qualityScore = validateQuality(generatedText);

if (qualityScore < 0.8) {
  // Retry generation or flag for manual review
  return [{
    json: {
      action: "retry",
      reason: "Quality below threshold",
      score: qualityScore
    }
  }];
}

return [{ json: { approved: true, score: qualityScore } }];
```

## Troubleshooting

### Common Issues

#### 1. CUDA Memory Issues
**Problem**: Out of memory during training/inference
**Solutions**:
```bash
# Check GPU memory
nvidia-smi

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size
--batch_size 1

# Enable gradient checkpointing
--use_gradient_checkpointing true

# Use 4-bit quantization
--load_in_4bit true
```

#### 2. Pickle Errors
**Problem**: `Can't pickle local function` error
**Solution**:
```python
# Set dataloader workers to 0
dataloader_num_workers = 0

# Or define functions globally instead of locally
def collate_fn(batch):
    # Global function definition
    pass
```

#### 3. Model Loading Issues
**Problem**: Model fails to load from checkpoint
**Solutions**:
```python
# Check checkpoint directory structure
ls -la models/qwen_reasoning/checkpoint-*/

# Load with specific device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load adapter separately if using LoRA
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
```

#### 4. Data Format Issues
**Problem**: Invalid data format causing training errors
**Solutions**:
```python
# Validate data format
python scripts/utils/validate_data.py --file data/training_data.jsonl

# Clean data
python scripts/utils/clean_data.py --input data/raw_data.jsonl --output data/cleaned_data.jsonl

# Check encoding
file -bi data/training_data.jsonl
# Should show: text/plain; charset=utf-8
```

### Performance Optimization

#### Training Speed
```bash
# Use mixed precision
--fp16 true

# Optimize dataloader
--dataloader_num_workers 0
--dataloader_pin_memory true

# Use compiled model (PyTorch 2.0+)
--torch_compile true
```

#### Inference Speed
```bash
# Use quantization
--load_in_8bit true

# Optimize generation parameters
--do_sample false
--num_beams 1
--max_length 1024  # Reduce if possible
```

#### Memory Optimization
```bash
# Enable gradient checkpointing
--use_gradient_checkpointing true

# Use DeepSpeed (for multi-GPU)
--deepspeed ds_config.json

# Clear cache regularly
python -c "import gc; gc.collect()"
```

## Advanced Usage

### Custom Model Training

#### 1. Custom Dataset Preparation
```python
# scripts/utils/prepare_custom_dataset.py
import json
import pandas as pd

def prepare_custom_dataset(source_file: str, output_file: str):
    """Convert custom format to training format"""
    
    # Load your custom data
    df = pd.read_csv(source_file)
    
    # Convert to required format
    training_data = []
    for _, row in df.iterrows():
        item = {
            "question": row['question'],
            "options": [
                f"A) {row['option_a']}",
                f"B) {row['option_b']}",
                f"C) {row['option_c']}",
                f"D) {row['option_d']}"
            ],
            "answer": row['correct_answer'],
            "topic": row.get('topic', 'General'),
            "difficulty": row.get('difficulty', 'Medium')
        }
        
        # Add reasoning chain if available
        if 'reasoning' in row and pd.notna(row['reasoning']):
            item['reasoning'] = row['reasoning']
            
        training_data.append(item)
    
    # Save as JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(training_data)} items to {output_file}")

# Usage
prepare_custom_dataset('my_data.csv', 'training_data.jsonl')
```

#### 2. Custom Training Configuration
```python
# scripts/training/custom_training_config.py
from transformers import TrainingArguments
from peft import LoraConfig

def get_custom_training_args():
    return TrainingArguments(
        output_dir="models/custom_model",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=25,
        eval_steps=250,
        save_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="wandb"
    )

def get_lora_config():
    return LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
```

### Model Evaluation

#### Comprehensive Evaluation Script
```python
# scripts/evaluation/comprehensive_eval.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model_path: str, test_file: str):
    """Comprehensive model evaluation"""
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load test data
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    predictions = []
    true_labels = []
    reasoning_quality = []
    
    for item in test_data:
        # Generate prediction
        prompt = format_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=2048,
                temperature=0.1,
                do_sample=False
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = extract_answer(response)
        
        predictions.append(predicted_answer)
        true_labels.append(item['answer'])
        
        # Evaluate reasoning quality
        if 'reasoning' in response:
            quality_score = evaluate_reasoning_quality(response)
            reasoning_quality.append(quality_score)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    avg_reasoning_quality = sum(reasoning_quality) / len(reasoning_quality)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'reasoning_quality': avg_reasoning_quality,
        'predictions': predictions
    }
```

### Deployment

#### API Server Setup
```python
# scripts/deployment/api_server.py
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load model once at startup
MODEL_PATH = "RayTsai/chinese-reasoning-qwen2.5-7b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

@app.route('/predict', methods=['POST'])
def predict():
    """Generate reasoning-based prediction"""
    data = request.json
    
    # Format prompt
    prompt = format_prompt(data)
    
    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=2048,
            temperature=0.1,
            do_sample=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = extract_answer(response)
    reasoning = extract_reasoning(response)
    
    return jsonify({
        'answer': answer,
        'reasoning': reasoning,
        'confidence': calculate_confidence(outputs)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### Docker Deployment
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python3", "scripts/deployment/api_server.py"]
```

This comprehensive usage guide covers all aspects of using the Kaggle #2 reasoning LLM system, from basic setup to advanced deployment scenarios. For additional help, refer to the specific README files in each directory or check the troubleshooting section above.