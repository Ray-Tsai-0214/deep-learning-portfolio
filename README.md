# Deep Learning Projects Portfolio

## 作品集簡介 | Portfolio Introduction

本作品集展示了三個深度學習專案，主要聚焦於大型語言模型（LLM）的微調和優化技術。這些專案源自 Kaggle 競賽，展現了從基礎微調到進階優化技術的完整學習歷程。

This portfolio showcases three deep learning projects focusing on Large Language Model (LLM) fine-tuning and optimization techniques. These projects originated from Kaggle competitions and demonstrate a complete learning journey from basic fine-tuning to advanced optimization techniques.

---

## 專案列表 | Project List

### [Project 1: LLM Fine-tuning Without Reasoning Information](./project-1-llm-finetuning-without-reasoning/)
**技術重點 | Technical Focus:**
- 基礎 LLM 微調技術
- Qwen2.5 模型的應用
- 中文資料集處理
- Basic LLM fine-tuning techniques
- Qwen2.5 model implementation
- Chinese dataset processing

**主要成果 | Key Achievements:**
- 成功實現 Qwen2.5-1M 模型的微調
- 建立完整的訓練和推論管線
- 產生競賽提交檔案

---

### [Project 2: LLM Fine-tuning With Reasoning Information](./project-2-llm-finetuning-with-reasoning/)
**技術重點 | Technical Focus:**
- 加入推理資訊的進階微調
- DeepSeek 和 Qwen 模型比較
- N8N 工作流程整合
- Advanced fine-tuning with reasoning information
- DeepSeek and Qwen model comparison
- N8N workflow integration

**主要成果 | Key Achievements:**
- 實現包含推理過程的資料增強
- 建立自動化資料準備管線
- 提升模型推理能力

---

### [Project 3: LLM GRPO Optimization](./project-3-llm-grpo-optimization/)
**技術重點 | Technical Focus:**
- GRPO (Group Relative Policy Optimization) 技術
- 強化學習優化方法
- 中文資料集 50% 訓練策略
- GRPO optimization techniques
- Reinforcement learning approaches
- Chinese dataset 50% training strategy

**主要成果 | Key Achievements:**
- 實現 GRPO 訓練流程
- 整合 WandB 監控系統
- 優化模型效能指標

---

## 技術棧 | Technology Stack

- **深度學習框架 | Deep Learning Frameworks:** PyTorch, Transformers (Hugging Face)
- **模型 | Models:** Qwen2.5, DeepSeek
- **優化技術 | Optimization:** LoRA, QLoRA, GRPO
- **工具 | Tools:** WandB, N8N, Git
- **程式語言 | Programming Languages:** Python

---

## 專案架構 | Project Structure

```
deep_learning/
│
├── project-1-llm-finetuning-without-reasoning/
│   ├── data/                    # 訓練和測試資料
│   ├── final_method/            # 最終實作方法
│   └── testing_method/          # 實驗測試方法
│
├── project-2-llm-finetuning-with-reasoning/
│   ├── data/                    # 包含推理資訊的資料集
│   ├── scripts/                 # 訓練和推論腳本
│   ├── n8n_workflow/           # 自動化工作流程
│   └── docs/                   # 技術文件
│
└── project-3-llm-grpo-optimization/
    ├── data/                    # GRPO 訓練資料
    ├── scripts/                 # GRPO 實作腳本
    └── images/                  # 訓練視覺化結果
```

---

## 學習重點 | Learning Highlights

1. **模型微調技術演進**：從基礎微調到加入推理資訊，再到使用 GRPO 優化
2. **實務經驗累積**：處理真實 Kaggle 競賽資料，面對實際挑戰
3. **工具鏈整合**：學習整合多種工具提升開發效率
4. **實驗管理**：建立系統化的實驗記錄和版本控制

---

## 未來發展 | Future Development

- 探索更多先進的 LLM 優化技術
- 擴展到多語言模型訓練
- 研究模型壓縮和部署優化
- 開發端到端的 MLOps 管線

---

## 聯絡資訊 | Contact Information

如有任何問題或合作機會，歡迎聯繫！

For any questions or collaboration opportunities, feel free to reach out!

---

## 授權 | License

本作品集中的專案僅供學習和展示用途。


Projects in this portfolio are for educational and demonstration purposes only.
