# Qwen 2.5 推理訓練系統配置

## 訓練配置

### 基礎設置
SELECTED_MODEL = "qwen2.5_1m"  # 可選: qwen2.5_1m, qwen2.5_7b, deepseek_qwen-14B
TRAINING_MODE = "reasoning"    # 可選: reasoning, simple, mixed, step_by_step
USE_4BIT = True               # 是否使用4-bit量化

### 數據路徑
TRAIN_FILE = "C:/Users/NTHUILST/Ray/DL/data/training_data_improve.csv"  # 訓練數據路徑，需包含推理過程
TEST_FILE = "../data/test-v2.csv" # 測試數據路徑

### 訓練參數
TRAINING_CONFIG = {
    "num_train_epochs": 4,
    "per_device_train_batch_size": 2,  # 推理訓練建議使用較小批次
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 1e-4,  # 推理訓練建議較低學習率
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "eval_steps": 50,
    "save_steps": 100,
    "logging_steps": 10
}

### LoRA配置
LORA_CONFIG = {
    "r": 32,          # LoRA秩，推理訓練建議較高
    "lora_alpha": 64, # LoRA縮放因子
    "lora_dropout": 0.1,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ]
}

### 序列長度
MAX_INPUT_LENGTH = 1024  # 輸入序列最大長度（推理需要更長）
MAX_OUTPUT_LENGTH = 512  # 輸出序列最大長度

## 推理配置

### 推理模式
REASONING_MODE = "full"  # 可選: full, concise, simple

### 生成配置
GENERATION_CONFIG = {
    "full": {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.1
    },
    "concise": {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "repetition_penalty": 1.1
    }
