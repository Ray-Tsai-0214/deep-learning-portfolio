# Data Directory

This directory contains sample training data used for the Kaggle #2 competition.

## Files

### `sample_train_data.jsonl`
- **Format**: JSON Lines
- **Content**: Basic training data without reasoning chains
- **Usage**: Baseline model training and comparison

### `sample_train_data_reasoning.jsonl`
- **Format**: JSON Lines  
- **Content**: Enhanced training data with structured reasoning chains
- **Usage**: Primary training data for reasoning-enabled models
- **Structure**: Each entry contains XML-formatted reasoning chains

## Data Format

### Standard Training Data
```json
{
  "question": "Question text in Chinese",
  "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
  "answer": "A",
  "topic": "政治治理",
  "difficulty": "中級"
}
```

### Reasoning Training Data
```json
{
  "question": "Question text in Chinese",
  "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
  "reasoning": "<question>...</question><think>...</think><reasoning>...</reasoning><reflection>...</reflection><adjustment>...</adjustment><output>A</output>",
  "answer": "A",
  "topic": "政治治理",
  "difficulty": "中級"
}
```

## Data Generation

The training data was generated using our automated N8N workflow system, which systematically covers:

- **15 Sensitive Topics**: Political governance, human rights, environmental protection, etc.
- **3 Difficulty Levels**: Basic, intermediate, advanced
- **5 Question Types**: Factual, conceptual, analytical, comparative, evaluative

## Usage Notes

- Use `sample_train_data_reasoning.jsonl` for training reasoning-capable models
- Use `sample_train_data.jsonl` for baseline comparisons
- The reasoning chains follow XML structure for consistency and parseability
- All questions focus on Chinese political and sensitive topics requiring neutral, fact-based responses