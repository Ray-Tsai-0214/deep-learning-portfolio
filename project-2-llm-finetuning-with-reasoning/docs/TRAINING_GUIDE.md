# Kaggle #2 訓練指南

## 環境設置

1. 安裝依賴：
```bash
pip install -r requirements.txt
```

2. 設置GPU環境：
```bash
export CUDA_VISIBLE_DEVICES=0
```

## 數據準備

1. **推理鏈數據生成**：
   - 使用GPT-4或Claude生成推理過程
   - 格式：問題 → 推理步驟 → 答案

2. **數據格式化**：
```bash
python scripts/clean_data.py
```

## 模型訓練

### 方案1：Qwen2.5-7B with Reasoning
```bash
python scripts/qwen_finetune_reasoning.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --train_data "data/sample_train_data_reasoning.jsonl" \
    --output_dir "./results_reasoning"
```

### 方案2：Qwen2.5-14B without Reasoning
```bash
python scripts/qwen_finetune_without_reasoning.py \
    --model_name "Qwen/Qwen2.5-14B-Instruct" \
    --train_data "data/sample_train_data.jsonl" \
    --output_dir "./results_simple"
```

### 方案3：DeepSeek-R1 Mixed
```bash
python scripts/deepseek_finetune_reasoning.py \
    --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" \
    --train_data "data/deepseek_train_data_mixed.jsonl" \
    --output_dir "./results_deepseek"
```

## 推理與提交

1. **生成預測**：
```bash
python scripts/predict_test_reasoning_data_qwen2.5.py \
    --model_path "./results_reasoning/checkpoint-best" \
    --test_file "../test-check-v2.csv" \
    --output_file "submission/submission.csv"
```

2. **驗證提交格式**：
```bash
python scripts/validate_submission.py submission/submission.csv
```

## 最佳實踐

1. **記憶體優化**：
   - 使用4-bit量化：`load_in_4bit=True`
   - 梯度檢查點：`gradient_checkpointing=True`
   - 批量大小調整：根據GPU記憶體動態調整

2. **訓練監控**：
   - 使用WandB追蹤訓練過程
   - 監控loss、學習率、GPU記憶體使用

3. **推理優化**：
   - 批量推理以提高效率
   - 使用`torch.no_grad()`減少記憶體使用
   - 實施答案提取的多重策略

## 故障排除

1. **CUDA OOM錯誤**：
   - 減少batch_size
   - 使用gradient_accumulation_steps
   - 啟用mixed precision training

2. **答案格式錯誤**：
   - 檢查prompt template
   - 使用regex提取答案
   - 實施fallback策略

## 競賽結果

- Public Score: 0.47177
- Private Score: 0.72043
- Rank: #30

這證明了推理鏈方法在處理複雜選擇題時的有效性。