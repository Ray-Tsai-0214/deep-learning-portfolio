# 訓練數據說明

## 主要數據檔案

### training_reasoning_data_chi.tsv
- **描述**: 中文推理訓練數據 (50%子集)
- **格式**: TSV (Tab-Separated Values)
- **大小**: 12,238個preference pairs
- **用途**: GRPO訓練的核心數據

### 數據格式

```tsv
question	option_A	option_B	option_C	option_D	correct_answer	reasoning
問題內容	選項A內容	選項B內容	選項C內容	選項D內容	A	詳細推理過程...
```

### 數據特色

1. **敏感議題覆蓋**: 包含15個敏感政治和社會議題
2. **推理鏈完整**: 每個問題都包含step-by-step推理過程
3. **中立性導向**: 推理過程強調客觀性和多角度思考
4. **格式統一**: 嚴格的數據格式確保訓練穩定性

### 主題分布

- 科技倫理與數字監管
- 經濟發展與貧富差距
- 環境保護與生態政策
- 教育體系與思想培養
- 網絡安全與數據主權
- 法律制度與司法獨立
- 人權議題與普世價值
- 政治與政府治理
- 言論自由與信息傳播
- 宗教與文化多樣性
- 民族關係與文化認同
- 地區自治與主權問題
- 國際關係與地緣政治
- 社會問題與公民權利
- 現代與歷史事件分析

## 數據預處理

### 清理步驟
1. 去除重複問題
2. 統一格式標準化
3. 推理鏈質量檢查
4. 中立性評估

### 質量控制
- 人工審核敏感內容
- 自動化格式驗證
- 推理邏輯一致性檢查
- 答案準確性確認

## 使用注意事項

1. **版權**: 數據僅供學術研究使用
2. **隱私**: 不包含個人隱私信息
3. **合規**: 遵守相關法律法規
4. **倫理**: 促進AI系統的公平性和中立性

## 數據生成流程

此數據集是通過N8N自動化工作流程生成，詳細過程請參考：
- [Kaggle #2 N8N工作流程](https://github.com/Deep-Learning-NYCU/kaggle-2-reasoning-llm-sft-with-reasoning-information-Ray-Tsai-0214)

## 相關論文和引用

如使用此數據集，請引用：

```bibtex
@misc{chinese-reasoning-dataset-2025,
  title={Chinese Reasoning Dataset for Sensitive Topics},
  author={Ray Tsai},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Deep-Learning-NYCU/kaggle-3-reasoning-llm-sft-grpo-Ray-Tsai-0214}
}
```