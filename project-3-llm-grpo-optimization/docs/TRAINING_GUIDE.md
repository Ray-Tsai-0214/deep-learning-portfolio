# GRPO訓練完整指南

## 概述

本指南詳細說明如何使用GRPO (Group Relative Policy Optimization) 方法訓練中文大語言模型，專門針對敏感政治議題的推理能力優化。

## 環境準備

### 1. 硬體需求

**推薦配置**:
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- CPU: 16+ 核心
- RAM: 32GB+
- 存儲: 500GB+ SSD

**最低配置**:
- GPU: RTX 4070 (16GB VRAM)
- CPU: 8+ 核心
- RAM: 16GB+
- 存儲: 200GB+ SSD

### 2. 軟體環境

```bash
# 創建虛擬環境
conda create -n grpo_training python=3.9
conda activate grpo_training

# 安裝CUDA (如需要)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia

# 安裝依賴
pip install -r requirements.txt
```

### 3. 環境變數設置

```bash
# WandB設置 (可選)
export WANDB_PROJECT="chinese-reasoning-grpo"
export WANDB_API_KEY="your_wandb_key"

# HuggingFace設置 (可選)
export HUGGINGFACE_TOKEN="your_hf_token"

# 避免tokenizer警告
export TOKENIZERS_PARALLELISM=false
```

## 數據準備

### 1. 訓練數據格式

數據應為TSV格式，包含以下欄位：

```tsv
question	option_A	option_B	option_C	option_D	correct_answer	reasoning
問題內容	選項A	選項B	選項C	選項D	A	詳細推理過程
```

### 2. 數據質量要求

- **完整性**: 所有欄位都必須填寫
- **一致性**: 正確答案必須是A、B、C、D之一
- **推理鏈**: reasoning欄位包含詳細的step-by-step分析
- **中立性**: 推理過程保持客觀中立

### 3. 數據預處理

```python
import pandas as pd

# 載入數據
df = pd.read_csv('training_data.tsv', sep='\t')

# 基本檢查
print(f"數據量: {len(df)}")
print(f"欄位: {df.columns.tolist()}")
print(f"答案分布: {df['correct_answer'].value_counts()}")

# 清理數據
df = df.dropna()  # 移除空值
df = df[df['correct_answer'].isin(['A', 'B', 'C', 'D'])]  # 確保答案有效
```

## GRPO訓練流程

### 1. 基本訓練指令

```bash
# 基本訓練
python scripts/grpo_training_chinese_50percent.py

# 指定GPU
CUDA_VISIBLE_DEVICES=0 python scripts/grpo_training_chinese_50percent.py

# 後台運行
nohup python scripts/grpo_training_chinese_50percent.py > training.log 2>&1 &
```

### 2. 訓練配置調整

主要配置參數位於訓練腳本中：

```python
grpo_config = GRPOConfig(
    learning_rate=3e-05,           # 學習率
    per_device_train_batch_size=16, # 批量大小
    gradient_accumulation_steps=2,  # 梯度累積
    num_train_epochs=2,            # 訓練輪數
    max_length=1024,               # 最大序列長度
    max_prompt_length=512,         # 最大提示長度
    dataloader_num_workers=0,      # 🔑關鍵: 避免Pickle錯誤
    save_steps=200,                # 保存步數
    logging_steps=10,              # 日誌步數
    bf16=True,                     # 混合精度
    gradient_checkpointing=True    # 梯度檢查點
)
```

### 3. 記憶體優化技巧

**4-bit量化配置**:
```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
```

**LoRA配置**:
```python
lora_config = LoraConfig(
    r=16,                          # LoRA秩
    lora_alpha=32,                 # 縮放參數
    lora_dropout=0.05,             # Dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"]
)
```

## 獎勵函數設計

### 1. 核心設計原則

GRPO的核心是獎勵函數，我們的設計考慮：

1. **答案格式正確性**
2. **推理過程完整性**
3. **中立性表達**
4. **內容質量**

### 2. 獎勵函數實現

```python
def global_reward_function(prompts, completions):
    rewards = []
    
    for completion in completions:
        reward = 0.5  # 基礎分數
        
        # 格式檢查 (+0.3)
        if any(marker in completion for marker in ["答案：", "Answer:"]):
            reward += 0.3
        
        # 推理檢查 (+0.2)
        if any(marker in completion for marker in ["理由：", "因為", "根據"]):
            reward += 0.2
        
        # 中立性檢查 (+0.1)
        neutral_words = ["可能", "相對", "不同觀點", "平衡"]
        bias_words = ["絕對", "必須", "唯一", "錯誤"]
        
        neutral_score = sum(1 for word in neutral_words if word in completion)
        bias_score = sum(1 for word in bias_words if word in completion)
        
        if neutral_score > bias_score:
            reward += 0.1
        
        # 答案有效性 (最低0.7)
        if extract_answer(completion) in ['A', 'B', 'C', 'D']:
            reward = max(reward, 0.7)
        
        rewards.append(min(1.0, reward))
    
    return torch.tensor(rewards)
```

### 3. 獎勵函數調試

```python
# 測試獎勵函數
test_completions = [
    "答案：A\n理由：根據相關資料分析...",
    "選擇B，因為這是唯一正確的答案",
    "A",
    "這個問題很複雜，需要考慮多個觀點..."
]

rewards = global_reward_function([], test_completions)
for i, (completion, reward) in enumerate(zip(test_completions, rewards)):
    print(f"回答 {i+1}: {reward:.2f}")
    print(f"內容: {completion[:50]}...")
    print()
```

## 訓練監控

### 1. WandB監控

訓練過程會自動記錄到WandB：

- **Loss曲線**: 訓練損失變化
- **Reward分布**: 獎勵分數統計
- **KL散度**: 與基礎模型的差異
- **GPU使用率**: 硬體監控

### 2. 本地日誌

```bash
# 查看訓練日誌
tail -f logs/grpo_training_*.log

# 監控GPU使用
watch -n 1 nvidia-smi

# 檢查模型檢查點
ls -la models/grpo_chinese_*/checkpoint-*/
```

### 3. 訓練指標解讀

**正常訓練指標**:
- Loss: 從0.5-1.0逐漸下降到0.05-0.1
- Reward: 從0.6-0.7逐漸提升到0.65-0.7
- KL散度: 保持在1-3之間

**異常情況處理**:
- Loss不下降: 檢查學習率、數據質量
- Reward不提升: 檢查獎勵函數邏輯
- KL散度過大: 可能需要調整KL懲罰係數

## 常見問題解決

### 1. Pickle錯誤

**問題**: `pickle.PicklingError` 或多進程錯誤

**解決**: 設置 `dataloader_num_workers=0`

```python
grpo_config = GRPOConfig(
    dataloader_num_workers=0,  # 關鍵設置
    # 其他配置...
)
```

### 2. CUDA OOM (顯存不足)

**解決方案**:

1. **減少批量大小**:
```python
per_device_train_batch_size=8  # 從16降到8
gradient_accumulation_steps=4   # 對應增加
```

2. **啟用梯度檢查點**:
```python
gradient_checkpointing=True
```

3. **使用更激進的量化**:
```python
# 考慮8-bit量化
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
```

### 3. 訓練中斷恢復

```python
# 從檢查點恢復
trainer = GRPOTrainer(...)
trainer.train(resume_from_checkpoint="path/to/checkpoint-xxx")
```

### 4. 模型收斂慢

**調整策略**:

1. **增加學習率**:
```python
learning_rate=5e-05  # 從3e-05增加
```

2. **調整獎勵函數**:
- 增加獎勵差異
- 簡化評估邏輯

3. **數據質量檢查**:
- 確保推理鏈質量
- 檢查答案分布平衡

## 最佳實踐

### 1. 訓練策略

1. **漸進式訓練**: 先用小數據集驗證，再用完整數據
2. **定期保存**: 每200步保存檢查點
3. **多次實驗**: 嘗試不同超參數組合
4. **監控對比**: 同時跟蹤多個指標

### 2. 硬體優化

1. **記憶體管理**: 定期清理GPU緩存
2. **溫度監控**: 確保GPU溫度正常
3. **電源管理**: 使用穩定的電源供應

### 3. 數據管理

1. **備份策略**: 定期備份訓練數據和模型
2. **版本控制**: 記錄數據和代碼版本
3. **質量監控**: 持續評估生成質量

## 結果評估

### 1. 自動評估

```bash
# 生成測試提交
python scripts/grpo_test_submission.py

# 檢查答案分布
python -c "
import pandas as pd
df = pd.read_csv('submission/submission_grpo_*.csv')
print(df['answer'].value_counts())
"
```

### 2. 質量檢查

```python
# 人工抽檢
import random
results = pd.read_csv('submission/detailed_results_grpo_*.csv')
sample = results.sample(10)

for _, row in sample.iterrows():
    print(f"問題 {row['id']}:")
    print(f"答案: {row['answer']}")
    print(f"回答: {row['response'][:200]}...")
    print("-" * 50)
```

### 3. 性能對比

對比不同階段的模型性能：

| 指標 | Kaggle #1 | Kaggle #2 | Kaggle #3 |
|------|-----------|-----------|-----------|
| 訓練時間 | 38分鐘 | 75分鐘 | 2400分鐘 |
| 記憶體使用 | 20GB | 18-22GB | 22GB |
| 最終分數 | 0.62 | Rank #30 | Reward 0.66 |

## 結論

GRPO訓練是一個複雜但強大的方法，通過精心設計的獎勵函數和適當的工程優化，可以顯著提升模型在敏感議題上的推理能力和中立性。

關鍵成功因素：
1. 高質量的推理數據
2. 合理的獎勵函數設計
3. 穩定的工程實現
4. 充分的訓練時間

通過本指南的實踐，您應該能夠成功復現我們的GRPO訓練結果，並進一步優化模型性能。