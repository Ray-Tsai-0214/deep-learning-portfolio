#!/usr/bin/env python3
"""
HuggingFace模型上傳腳本
將訓練完成的GRPO模型上傳到HuggingFace Hub
"""

import os
import json
from datetime import datetime
from huggingface_hub import HfApi, login, create_repo, upload_folder

# HuggingFace token (請設置環境變數或直接填入)
TOKEN = os.getenv("HUGGINGFACE_TOKEN", "your_token_here")

def create_model_card():
    """創建模型卡片"""
    return """---
language:
- zh
license: apache-2.0
tags:
- chinese
- llm
- lora
- qwen2.5
- grpo
- reinforcement-learning
base_model: Qwen/Qwen2.5-7B-Instruct
datasets:
- custom-chinese-reasoning
model-index:
- name: chinese-grpo-qwen2.5-7b-50percent
  results:
    - task:
        type: text-generation
      metrics:
        - name: Reward Score
          type: reward
          value: 0.66
        - name: Training Loss
          type: loss
          value: 0.058
---

# Chinese GRPO Qwen2.5-7B (50% Dataset)

使用GRPO (Group Relative Policy Optimization)方法訓練的中文大語言模型，專門優化處理敏感議題的中立性和推理能力。

## 模型詳情

- **基礎模型**: Qwen/Qwen2.5-7B-Instruct
- **訓練方法**: GRPO with LoRA
- **訓練數據**: 50%中文推理數據集 (12,238個preference pairs)
- **訓練時間**: 39小時44分鐘
- **硬體**: NVIDIA RTX 4090 24GB
- **最終獎勵分數**: 0.66

## 訓練配置

```yaml
LoRA配置:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

訓練參數:
  learning_rate: 3e-05
  batch_size: 16
  gradient_accumulation_steps: 2
  num_epochs: 2
  total_steps: 5,506
  
優化設置:
  quantization: 4-bit
  gradient_checkpointing: true
  bf16: true
  dataloader_num_workers: 0  # 解決pickle錯誤
```

## 性能指標

- **最終訓練損失**: 0.058
- **獎勵分數**: 0.6604 ± 0.068
- **KL散度**: 1.90
- **處理tokens**: 65,428,674

## 使用方法

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# 量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 載入基礎模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
    trust_remote_code=True
)

# 載入LoRA權重
model = PeftModel.from_pretrained(
    base_model,
    "RayTsai/chinese-grpo-qwen2.5-7b-50percent"
)

# 載入tokenizer
tokenizer = AutoTokenizer.from_pretrained("RayTsai/chinese-grpo-qwen2.5-7b-50percent")

# 使用模型
prompt = "問題：[您的問題]\\n\\n選項：\\nA. [選項A]\\nB. [選項B]\\nC. [選項C]\\nD. [選項D]\\n\\n請選擇正確答案並說明理由。"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 特色功能

1. **中立性優化**: 通過GRPO訓練提升回答的客觀性
2. **推理能力**: 不僅給出答案，還提供詳細推理過程
3. **穩定性**: 40小時訓練確保模型收斂
4. **效率**: 4-bit量化支援在消費級GPU運行

## 訓練日誌

- 開始時間: 2024-06-24 22:31:39
- 結束時間: 2024-06-26 14:19:35
- 總步數: 5,506/5,508 (99.96%)
- 保存檢查點: 27個（每200步）

## 技術創新

1. **GRPO在中文敏感議題的首次應用**
2. **中立性感知的獎勵函數設計**
3. **解決Pickle錯誤的工程優化**
4. **長時間穩定訓練的實現**

## 引用

```bibtex
@misc{chinese-grpo-2025,
  author = {Ray Tsai},
  title = {Chinese GRPO Qwen2.5-7B 50% Dataset},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/RayTsai/chinese-grpo-qwen2.5-7b-50percent}
}
```

## 授權

Apache License 2.0

## 聯繫方式

如有問題或合作意向，請通過HuggingFace平台聯繫。

---

**免責聲明**: 此模型用於學術研究目的，生成的內容不代表作者觀點。使用時請遵守相關法規。
"""

def upload_model():
    """上傳模型到HuggingFace"""
    print("🚀 開始上傳GRPO模型到HuggingFace...")
    
    if TOKEN == "your_token_here":
        print("❌ 請先設置HuggingFace token!")
        print("方法1: 設置環境變數 HUGGINGFACE_TOKEN")
        print("方法2: 直接在腳本中修改TOKEN變數")
        return
    
    # 登入
    login(token=TOKEN, add_to_git_credential=False)
    api = HfApi()
    
    # 模型路徑和名稱
    model_path = "../models/grpo_chinese_50percent_0624/final_model"
    repo_name = "chinese-grpo-qwen2.5-7b-50percent"
    repo_id = f"RayTsai/{repo_name}"
    
    print(f"📦 模型路徑: {model_path}")
    print(f"📝 目標repo: {repo_id}")
    
    # 檢查模型路徑
    if not os.path.exists(model_path):
        print(f"❌ 模型路徑不存在: {model_path}")
        print("請確保GRPO訓練已完成並保存了模型")
        return
    
    try:
        # 創建或更新repo
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print("✅ Repository已準備就緒")
        
        # 創建README
        readme_path = os.path.join(model_path, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(create_model_card())
        print("✅ 模型卡片已創建")
        
        # 上傳所有檔案
        print("📤 開始上傳檔案...")
        upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload GRPO fine-tuned model (50% Chinese dataset, 40hrs training)"
        )
        
        print("✅ 上傳成功！")
        print(f"🔗 模型連結: https://huggingface.co/{repo_id}")
        
        # 保存上傳記錄
        upload_log = {
            "upload_time": datetime.now().isoformat(),
            "repo_id": repo_id,
            "model_path": model_path,
            "status": "success"
        }
        
        log_path = "../logs/huggingface_upload.json"
        os.makedirs("../logs", exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(upload_log, f, ensure_ascii=False, indent=2)
        
        print(f"📋 上傳記錄已保存: {log_path}")
        
        # 清理臨時README
        if os.path.exists(readme_path):
            os.remove(readme_path)
            
    except Exception as e:
        print(f"❌ 上傳失敗: {e}")
        
        # 保存錯誤記錄
        error_log = {
            "upload_time": datetime.now().isoformat(),
            "repo_id": repo_id,
            "model_path": model_path,
            "status": "failed",
            "error": str(e)
        }
        
        log_path = "../logs/huggingface_upload_error.json"
        os.makedirs("../logs", exist_ok=True)
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(error_log, f, ensure_ascii=False, indent=2)
        
        raise

def test_connection():
    """測試HuggingFace連接"""
    print("🔍 測試HuggingFace連接...")
    
    if TOKEN == "your_token_here":
        print("❌ 請先設置HuggingFace token!")
        return False
    
    try:
        login(token=TOKEN, add_to_git_credential=False)
        api = HfApi()
        user_info = api.whoami()
        
        print("✅ 連接成功！")
        print(f"👤 用戶名: {user_info.get('name', 'N/A')}")
        print(f"📧 Email: {user_info.get('email', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 連接失敗: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🤗 HuggingFace 模型上傳工具")
    print("=" * 50)
    
    # 測試連接
    if test_connection():
        print("\n" + "=" * 50)
        upload_model()
    else:
        print("\n❌ 請檢查token設置後重試")