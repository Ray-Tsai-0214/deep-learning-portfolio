# Utility Scripts

This directory contains helper functions and configuration utilities used across the project.

## Scripts Overview

### `clean_data.py`
- **Purpose**: Data preprocessing and cleaning
- **Functions**:
  - Text normalization for Chinese content
  - Duplicate detection and removal
  - Format validation and correction
  - Quality filtering based on criteria

### `config.py`
- **Purpose**: Centralized configuration management
- **Features**:
  - Model parameters and hyperparameters
  - Training configuration settings
  - File paths and directory structure
  - Hardware optimization settings

## Functions

### Data Cleaning (`clean_data.py`)

#### `normalize_chinese_text(text)`
- Standardizes Chinese character encoding
- Removes unnecessary whitespace and punctuation
- Handles traditional/simplified Chinese conversion

#### `validate_reasoning_format(reasoning_text)`
- Validates XML structure of reasoning chains
- Ensures all required tags are present
- Checks for proper nesting and formatting

#### `remove_duplicates(dataset)`
- Identifies and removes duplicate questions
- Preserves highest quality versions
- Maintains dataset balance across topics

#### `filter_by_quality(dataset, threshold=0.8)`
- Filters data based on quality metrics
- Removes low-confidence generations
- Ensures neutrality standards compliance

### Configuration Management (`config.py`)

#### Model Configuration
```python
MODEL_CONFIG = {
    'qwen': {
        'name': 'Qwen/Qwen2.5-7B-Instruct',
        'max_length': 2048,
        'temperature': 0.1,
        'top_p': 0.9
    },
    'deepseek': {
        'name': 'deepseek-ai/deepseek-r1-14b',
        'max_length': 4096,
        'temperature': 0.2,
        'top_p': 0.95
    }
}
```

#### Training Configuration
```python
TRAINING_CONFIG = {
    'batch_size': 4,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 100,
    'weight_decay': 0.01,
    'gradient_clipping': 1.0
}
```

#### Hardware Configuration
```python
HARDWARE_CONFIG = {
    'use_gpu': True,
    'mixed_precision': True,
    'gradient_checkpointing': True,
    'dataloader_num_workers': 0  # For pickle compatibility
}
```

## Usage Examples

### Data Cleaning
```python
from scripts.utils.clean_data import normalize_chinese_text, validate_reasoning_format

# Clean text data
cleaned_text = normalize_chinese_text(raw_text)

# Validate reasoning chain format
is_valid = validate_reasoning_format(reasoning_chain)
```

### Configuration Loading
```python
from scripts.utils.config import MODEL_CONFIG, TRAINING_CONFIG

# Load model configuration
model_name = MODEL_CONFIG['qwen']['name']
max_length = MODEL_CONFIG['qwen']['max_length']

# Load training configuration
batch_size = TRAINING_CONFIG['batch_size']
learning_rate = TRAINING_CONFIG['learning_rate']
```

## Quality Assurance

### Text Quality Metrics
- **Readability score**: Chinese text complexity assessment
- **Neutrality score**: Political bias detection
- **Coherence score**: Logical consistency evaluation
- **Completeness score**: Information coverage assessment

### Reasoning Chain Validation
- **XML structure**: Proper tag nesting and closure
- **Content completeness**: All required sections present
- **Logical flow**: Step-by-step reasoning coherence
- **Neutrality check**: Balanced perspective verification

## Performance Utilities

### Memory Management
```python
def optimize_memory():
    """Optimize memory usage for large model training"""
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
```

### Progress Tracking
```python
def setup_logging(log_file):
    """Configure logging for training and inference"""
    import logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
```

## Integration Points

These utilities are used throughout the project:
- **Training scripts** use config and data cleaning functions
- **Inference scripts** leverage quality validation
- **Data generation** workflows use cleaning and validation
- **Evaluation scripts** employ quality metrics

## Best Practices

- **Modular design**: Each function has a single responsibility
- **Error handling**: Comprehensive exception management
- **Documentation**: Clear docstrings and examples
- **Testing**: Unit tests for critical functions