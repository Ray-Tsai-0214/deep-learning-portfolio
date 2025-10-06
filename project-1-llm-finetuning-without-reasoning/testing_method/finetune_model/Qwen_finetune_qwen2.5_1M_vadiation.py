#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import json
import numpy as np
import logging
import time
import random
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from huggingface_hub import login
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 配置日誌系統
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("finetune.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 確保CUDA可用
assert torch.cuda.is_available(), "需要CUDA支持"
logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
logger.info(f"顯存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 中文大語言模型列表
CHINESE_MODELS = {
    "qwen2_7b": "Qwen/Qwen2-7B",
    "qwen_QwQ": "Qwen/QwQ-32B",
    "qwen1.5_7b": "Qwen/Qwen1.5-7B",
    "qwen2.5_1m": "Qwen/Qwen2.5-14B-Instruct-1M",
    "qwen1.5_14b": "Qwen/Qwen1.5-14B",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek_qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek_qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek_lamma": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
}

# 記錄訓練進度的全局變數
class TrainingState:
    def __init__(self):
        self.best_accuracy = 0.0
        self.best_step = 0
        self.validation_results = []

training_state = TrainingState()

# 1. 準備數據 - 增強版，包含分層抽樣
def prepare_data(file_path, tokenizer=None, validation_method="stratified", validation_size=0.1, seed=42):
    """
    從CSV文件準備訓練數據，並使用適當方法分割驗證集
    
    參數:
        file_path: CSV文件路徑
        tokenizer: 用於文本處理的分詞器，若為None則僅格式化數據
        validation_method: 'stratified'(分層抽樣), 'random'(隨機抽樣)
        validation_size: 驗證集比例
        seed: 隨機種子
    """
    try:
        # 讀取CSV數據
        df = pd.read_csv(file_path)
        logger.info(f"CSV文件的欄位名稱: {df.columns.tolist()}")
        logger.info(f"讀取了 {len(df)} 筆訓練數據")
        
        # 欄位名稱映射
        question_col = 'question'
        option_a_col = 'option_A'
        option_b_col = 'option_B'
        option_c_col = 'option_C'
        option_d_col = 'option_D'
        answer_col = 'answer'
        
        # 處理可能的空值
        for col in [question_col, option_a_col, option_b_col, option_c_col, option_d_col]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)
        
        # 如果沒有answer列，可能是測試集
        is_test_file = answer_col not in df.columns
        
        # 分割訓練集和驗證集
        if not is_test_file:
            if validation_method == "stratified":
                try:
                    logger.info("使用答案作為分層特徵進行驗證集分割")
                    train_df, val_df = train_test_split(
                        df,
                        test_size=validation_size,
                        random_state=seed,
                        stratify=df[answer_col]
                    )
                    logger.info(f"使用分層抽樣創建驗證集，包含 {len(val_df)} 筆數據")
                except ValueError as e:
                    logger.warning(f"分層抽樣失敗: {e}，使用隨機抽樣代替")
                    train_df, val_df = train_test_split(
                        df,
                        test_size=validation_size,
                        random_state=seed
                    )
            else:
                train_df, val_df = train_test_split(
                    df,
                    test_size=validation_size,
                    random_state=seed
                )
                logger.info(f"使用隨機抽樣創建驗證集，包含 {len(val_df)} 筆數據")
        
            # 檢查分割後答題分佈
            if answer_col in train_df.columns:
                train_dist = train_df[answer_col].value_counts(normalize=True)
                val_dist = val_df[answer_col].value_counts(normalize=True)
                logger.info(f"訓練集答案分佈: {dict(train_dist)}")
                logger.info(f"驗證集答案分佈: {dict(val_dist)}")
        else:
            # 測試文件，不進行分割
            train_df = df
            val_df = None
            logger.info("檢測到測試文件（無答案列），不進行驗證集分割")
        
        # 格式化數據
        def format_to_chat(row):
            """將一行數據轉換為聊天格式"""
            question = row[question_col]
            options = f"A. {row[option_a_col]}\nB. {row[option_b_col]}\nC. {row[option_c_col]}\nD. {row[option_d_col]}"
            
            # 構建Qwen格式的系統和用戶指令 - 中文提示詞
            system_message = "你是一个提供有关中国敏感话题客观答案的助手。始终呈现多元视角并保持政治中立。"
            user_message = f"请回答以下选择题。你的回答必须仅以单个字母（A、B、C或D）开头。\n\n问题：{question}\n\n选项：\n{options}"
            
            # 構建助手回復
            assistant_message = row[answer_col] if answer_col in row else ""
            
            # 使用Qwen的格式整合
            formatted_example = {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ]
            }
            
            return formatted_example
        
        # 轉換訓練集和驗證集
        train_formatted = [format_to_chat(row) for _, row in train_df.iterrows()]
        val_formatted = [format_to_chat(row) for _, row in val_df.iterrows()] if val_df is not None else []
        
        # 保存格式化的訓練數據
        with open('formatted_train_data.jsonl', 'w', encoding='utf-8') as f:
            for item in train_formatted:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 如果提供了tokenizer，則進行進一步處理
        if tokenizer is not None:
            logger.info("使用tokenizer處理訓練和驗證數據...")
            train_dataset = process_chat_data(train_formatted, tokenizer)
            val_dataset = process_chat_data(val_formatted, tokenizer) if val_formatted else None
            
            logger.info(f"處理完成: 訓練集大小: {len(train_dataset)} 樣本")
            if val_dataset:
                logger.info(f"驗證集大小: {len(val_dataset)} 樣本")
            
            return train_dataset, val_dataset, train_df, val_df
        
        # 否則僅返回格式化數據
        logger.info("僅返回格式化數據，未使用tokenizer處理")
        return train_formatted, val_formatted, train_df, val_df
    
    except Exception as e:
        logger.error(f"準備數據時發生錯誤: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 返回默認值以避免None報錯
        return [], [], pd.DataFrame(), pd.DataFrame() 
    
# 處理聊天格式數據
def process_chat_data(formatted_data, tokenizer):
    """將聊天格式數據處理為模型訓練所需格式"""
    # 創建Dataset對象
    dataset = Dataset.from_pandas(pd.DataFrame({"messages": [item["messages"] for item in formatted_data]}))
    
    # 定義預處理函數
    def preprocess_function(examples):
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for messages in examples["messages"]:
            chat_text = ""
            labels_text = ""
            
            # 構建完整對話文本
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    chat_text += f"<|im_start|>system\n{message['content']}<|im_end|>\n"
                elif message["role"] == "user":
                    chat_text += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                    chat_text += "<|im_start|>assistant\n"
                elif message["role"] == "assistant" and i > 0:
                    labels_text = f"{message['content']}<|im_end|>"
            
            # 編碼輸入部分
            tokenized_input = tokenizer(
                chat_text, 
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors=None
            )
            
            # 編碼輸出標籤部分
            tokenized_labels = tokenizer(
                labels_text,
                truncation=True,
                max_length=48,
                padding=False,
                return_tensors=None
            )
            
            # 組合輸入和標籤
            input_ids = tokenized_input["input_ids"]
            combined_input_ids = input_ids + tokenized_labels["input_ids"]
            attention_mask = [1] * len(combined_input_ids)
            
            # 標籤: 輸入部分為-100，輸出部分為實際token ID
            labels = [-100] * len(input_ids) + tokenized_labels["input_ids"]
            
            all_input_ids.append(combined_input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels
        }
    
    # 應用預處理
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=min(8, os.cpu_count() or 1),
        desc="處理訓練數據"
    )
    
    return processed_dataset

# 2. 從Hugging Face下載並設置模型
def setup_model(model_name_or_key="qwen2.5_1m", use_4bit=True):
    """
    從Hugging Face下載並設置模型
    
    參數:
        model_name_or_key: 模型名稱（從CHINESE_MODELS字典中選擇）或直接使用Hugging Face模型ID
        use_4bit: 是否使用4-bit量化（推薦用於RTX 4090）
    """
    # 獲取模型ID
    if model_name_or_key in CHINESE_MODELS:
        model_id = CHINESE_MODELS[model_name_or_key]
    else:
        model_id = model_name_or_key  # 直接使用作為模型ID
    
    logger.info(f"正在從Hugging Face下載模型: {model_id}")
    
    # 量化配置
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    
    # 下載並加載tokenizer
    logger.info("下載tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        padding_side="left"
    )
    
    # 確保tokenizer有padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 下載並加載模型
    logger.info("下載模型（這可能需要幾分鐘時間）...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # 使用半精度加載
            attn_implementation="flash_attention_2"  # 啟用FlashAttention 2
        )
    except Exception as e:
        logger.error(f"模型下載失敗: {e}")
        logger.info("嘗試使用較大超時時間和降低記憶體使用重新下載...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder="offload",  # 如果RAM不足，可以使用磁碟卸載
            max_memory={0: "22GiB"}  # 限制GPU內存使用
        )
    
    logger.info("模型下載完成!")
    
    # 根據模型類型確定target_modules
    if "qwen" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "yi" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "deepseek" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "baichuan" in model_id.lower():
        target_modules = ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "internlm" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # 準備kbit訓練模型
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 應用LoRA配置
    model = get_peft_model(model, lora_config)
    
    # 打印可訓練參數
    trainable_params, all_params = model.get_nb_trainable_parameters()
    logger.info(f"可訓練參數: {trainable_params:,} ({trainable_params/all_params*100:.2f}% of {all_params:,})")
    
    return model, tokenizer

# 2. 評估指標計算 - 專為選擇題設計
def compute_metrics(eval_preds, tokenizer):
    """計算選擇題多項選擇的評估指標"""
    predictions, labels = eval_preds
    
    # 只取預測的最高值作為token ID
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=-1)
    
    # 解碼預測和標籤
    decoded_preds = []
    decoded_labels = []
    
    for pred_ids, label_ids in zip(predictions, labels):
        # 過濾掉padding和special tokens
        pred_ids = [id for id in pred_ids if id >= 0]
        label_ids = [id for id in label_ids if id >= 0 and id != -100]
        
        # 解碼
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True).strip()
        label_text = tokenizer.decode(label_ids, skip_special_tokens=True).strip()
        
        decoded_preds.append(pred_text)
        decoded_labels.append(label_text)
    
    # 提取選項字母（A, B, C, D）
    pred_letters = []
    label_letters = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        label_letter = next((c for c in label.upper() if c in "ABCD"), None)
        pred_letter = next((c for c in pred.upper() if c in "ABCD"), None)
        
        if label_letter:
            label_letters.append(label_letter)
            pred_letters.append(pred_letter if pred_letter else "NONE")
    
    # 計算準確率
    correct = sum(p == l for p, l in zip(pred_letters, label_letters))
    accuracy = correct / len(label_letters) if label_letters else 0
    
    # 計算每個選項的分布
    pred_dist = {}
    for letter in "ABCD":
        pred_dist[letter] = pred_letters.count(letter)
    pred_dist["NONE"] = pred_letters.count("NONE")
    
    # 計算混淆矩陣
    confusion = {}
    for pl, ll in zip(pred_letters, label_letters):
        key = f"{ll}->{pl}"
        confusion[key] = confusion.get(key, 0) + 1
    
    return {
        "accuracy": accuracy,
        "pred_distribution": pred_dist,
        "confusion": confusion
    }

# 4. 自定義訓練回調 - 增強驗證和報告
class ValidationReportCallback(TrainerCallback):
    """增強驗證報告回調"""
    def __init__(self, tokenizer, raw_val_df=None):
        self.tokenizer = tokenizer
        self.raw_val_df = raw_val_df  # 原始驗證數據，用於測試風格評估
        self.best_accuracy = 0.0
        self.best_step = 0
        self.best_model_path = None
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """每次評估後的回調"""
        if metrics:
            step = state.global_step
            accuracy = metrics.get("eval_accuracy", 0)
            
            # 記錄評估結果
            logger.info("\n" + "="*50)
            logger.info(f"步驟 {step} 的驗證結果:")
            logger.info(f"準確率: {accuracy:.4f}")
            
            # 記錄預測分布
            if "eval_pred_distribution" in metrics:
                dist = metrics["eval_pred_distribution"]
                logger.info("\n預測分布:")
                for key, value in dist.items():
                    logger.info(f"  {key}: {value}")
            
            # 記錄混淆矩陣
            if "eval_confusion" in metrics:
                confusion = metrics["eval_confusion"]
                logger.info("\n混淆矩陣:")
                for key, value in confusion.items():
                    logger.info(f"  {key}: {value}")
            
            # 如果有原始驗證數據，執行測試風格評估
            if self.raw_val_df is not None and hasattr(kwargs.get("model", None), "generate"):
                model = kwargs.get("model")
                sample_size = min(20, len(self.raw_val_df))  # 限制數量
                val_sample = self.raw_val_df.sample(sample_size)
                
                logger.info("\n執行測試風格評估...")
                test_style_metrics = self.test_style_evaluate(model, val_sample)
                
                # 記錄結果
                logger.info(f"\n測試風格評估結果 (樣本數: {sample_size}):")
                logger.info(f"準確率: {test_style_metrics['accuracy']:.4f}")
                logger.info("預測分布:")
                for key, value in test_style_metrics["distribution"].items():
                    logger.info(f"  {key}: {value}")
            
            # 檢查是否為最佳模型
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_step = step
                
                # 更新全局跟蹤器
                training_state.best_accuracy = accuracy
                training_state.best_step = step
                
                logger.info(f"\n發現新的最佳模型! 準確率: {accuracy:.4f}")
                
                # 如果未設置load_best_model_at_end，則保存最佳模型
                if not args.load_best_model_at_end and hasattr(kwargs.get("model", None), "save_pretrained"):
                    self.best_model_path = os.path.join(args.output_dir, "best_model")
                    kwargs["model"].save_pretrained(self.best_model_path)
                    logger.info(f"保存最佳模型到: {self.best_model_path}")
            
            # 保存驗證結果
            training_state.validation_results.append({
                "step": step,
                "accuracy": accuracy,
                "metrics": metrics
            })
            
            # 將完整評估結果保存到文件
            with open("validation_results.jsonl", "a", encoding="utf-8") as f:
                result_record = {
                    "step": step,
                    "accuracy": accuracy,
                    "distribution": metrics.get("eval_pred_distribution", {}),
                    "confusion": metrics.get("eval_confusion", {})
                }
                f.write(json.dumps(result_record, ensure_ascii=False) + "\n")
            
            logger.info("="*50)
    
    def test_style_evaluate(self, model, df_sample):
        """使用測試相同的方式進行評估"""
        model.eval()
        correct = 0
        total = 0
        predictions = {"A": 0, "B": 0, "C": 0, "D": 0, "NONE": 0}
        
        for _, row in df_sample.iterrows():
            # 創建推理提示詞
            question = row['question']
            options = f"A. {row['option_A']}\nB. {row['option_B']}\nC. {row['option_C']}\nD. {row['option_D']}"
            
            # 使用中文系統提示詞
            system_prompt = "你是一个提供有关中国敏感话题客观答案的助手。始终呈现多元视角并保持政治中立。"
            user_prompt = f"请回答以下选择题。你的回答必须仅以单个字母（A、B、C或D）开头。\n\n问题：{question}\n\n选项：\n{options}"
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"
            
            # 編碼輸入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解碼輸出
            generated_ids = outputs[0, inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # 提取答案選項
            pred_option = next((c for c in response.upper() if c in "ABCD"), None)
            
            if pred_option:
                predictions[pred_option] += 1
            else:
                predictions["NONE"] += 1
            
            # 檢查是否正確
            expected = row.get('answer')
            if expected and pred_option:
                if expected == pred_option:
                    correct += 1
                total += 1
        
        # 計算準確率
        accuracy = correct / total if total > 0 else 0
        
        return {
            "accuracy": accuracy,
            "distribution": predictions,
            "total": total,
            "correct": correct
        }

# 5. 訓練模型
def train_model(model, tokenizer, train_dataset, eval_dataset=None, raw_val_df=None, output_dir="./results"):
    """訓練模型，優化驗證和報告"""
    # 數據整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,  # 對於混合精度訓練很重要
    )
    
    # 為4090 GPU優化的訓練參數
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # 增加訓練輪數以提高學習效果
        per_device_train_batch_size=4,  # 基於4090的VRAM優化
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # 累積梯度以模擬更大批量
        evaluation_strategy="steps",
        eval_steps=100,  # 更頻繁的評估
        save_strategy="steps",
        save_steps=200,
        logging_steps=20,
        learning_rate=1e-4,  # 減小學習率，提高穩定性
        weight_decay=0.01,
        warmup_ratio=0.05,  # 增加預熱比例
        lr_scheduler_type="cosine_with_restarts",  # 使用重啟式余弦調度
        report_to="tensorboard",  # 使用tensorboard記錄訓練過程
        gradient_checkpointing=True,  # 啟用梯度檢查點以節省內存
        fp16=False,  # 禁用fp16以減少精度問題
        bf16=True,  # 如果GPU支持bf16，使用bfloat16
        optim="adamw_torch",
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        group_by_length=True,  # 可加速訓練
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # 使用準確率選擇最佳模型
        greater_is_better=True,
        save_total_limit=3  # 只保存最近的3個檢查點
    )
    
    # 定義自定義compute_metrics來傳入tokenizer
    def compute_metrics_wrapper(eval_preds):
        return compute_metrics(eval_preds, tokenizer)
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper if eval_dataset else None,
        callbacks=[
            ValidationReportCallback(tokenizer, raw_val_df),
            EarlyStoppingCallback(early_stopping_patience=5)  # 添加早停機制
        ]
    )
    
    # 開始訓練
    logger.info("開始訓練模型...")
    train_result = trainer.train()
    
    # 記錄訓練結果
    logger.info("訓練完成!")
    logger.info(f"總訓練步數: {train_result.global_step}")
    
    # 保存最終模型
    model_save_path = os.path.join(output_dir, "final_model")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"最終模型保存至: {model_save_path}")
    
    # 如果有驗證集，進行最終評估
    if eval_dataset:
        logger.info("進行最終評估...")
        metrics = trainer.evaluate()
        logger.info(f"最終評估結果: {metrics}")
    
    # 記錄最佳模型信息
    logger.info(f"最佳模型步驟: {training_state.best_step}")
    logger.info(f"最佳模型準確率: {training_state.best_accuracy:.4f}")
    
    return model, tokenizer, model_save_path

# 6. 獨立驗證函數
def validate_model(model_path, test_file, base_model_id=None):
    """使用測試風格方式驗證已訓練的模型"""
    logger.info(f"正在驗證模型: {model_path}")
    
    # 載入測試數據
    test_df = pd.read_csv(test_file)
    logger.info(f"測試集包含 {len(test_df)} 個問題")
    
    # 確定是否有答案列
    has_answers = "answer" in test_df.columns
    
    # 載入tokenizer
    if base_model_id is None:
        # 嘗試從模型路徑載入
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        # 使用基礎模型
        if base_model_id in CHINESE_MODELS:
            base_model_id = CHINESE_MODELS[base_model_id]
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    # 載入模型
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    except:
        logger.info("直接載入失敗，嘗試載入基礎模型和adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    
    model.eval()
    
    # 準備生成配置
    gen_config = GenerationConfig(
        max_new_tokens=10,
        do_sample=False,
        temperature=0.1,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 評估結果
    results = []
    correct = 0
    total = 0
    
    # 批次處理
    batch_size = 8
    progress_bar = tqdm(range(0, len(test_df), batch_size), desc="驗證進度")
    
    for i in progress_bar:
        batch_df = test_df.iloc[i:min(i + batch_size, len(test_df))]
        batch_prompts = []
        batch_answers = []
        
        for _, row in batch_df.iterrows():
            # 創建提示詞
            question = row['question']
            options = f"A. {row['option_A']}\nB. {row['option_B']}\nC. {row['option_C']}\nD. {row['option_D']}"
            
            # 使用中文系統提示詞
            system_prompt = "你是一个提供有关中国敏感话题客观答案的助手。始终呈现多元视角并保持政治中立。"
            user_prompt = f"请回答以下选择题。你的回答必须仅以单个字母（A、B、C或D）开头。\n\n问题：{question}\n\n选项：\n{options}"
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"
            
            batch_prompts.append(prompt)
            if has_answers:
                batch_answers.append(row['answer'])
            else:
                batch_answers.append(None)
        
        # 批次生成
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)
        
        # 處理結果
        for j, output_ids in enumerate(outputs):
            # 提取生成部分
            generated_ids = output_ids[inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # 提取答案選項
            pred_option = next((c for c in response.upper() if c in "ABCD"), None)
            
            # 記錄結果
            result = {
                "id": batch_df.iloc[j].get("ID", i + j),
                "question": batch_df.iloc[j]['question'],
                "expected": batch_answers[j],
                "predicted": pred_option,
                "response": response
            }
            
            results.append(result)
            
            # 計算準確率
            if batch_answers[j] and pred_option:
                if batch_answers[j] == pred_option:
                    correct += 1
                total += 1
    
    # 生成報告
    accuracy = correct / total if total > 0 else 0
    
    logger.info("\n" + "=" * 50)
    logger.info("驗證完成")
    logger.info(f"總問題數: {len(results)}")
    
    if has_answers:
        logger.info(f"可評估問題數: {total}")
        logger.info(f"正確數: {correct}")
        logger.info(f"準確率: {accuracy:.4f}")
    
    # 計算預測分布
    pred_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "無答案": 0}
    for result in results:
        pred = result.get("predicted")
        if pred in "ABCD":
            pred_dist[pred] += 1
        else:
            pred_dist["無答案"] += 1
    
    logger.info("\n預測分布:")
    for key, value in pred_dist.items():
        percentage = value / len(results) * 100
        logger.info(f"  {key}: {value} ({percentage:.1f}%)")
    
    # 如果有答案，顯示混淆矩陣
    if has_answers:
        confusion = {}
        for result in results:
            expected = result.get("expected")
            predicted = result.get("predicted", "無答案")
            if expected:
                key = f"{expected}->{predicted}"
                confusion[key] = confusion.get(key, 0) + 1
        
        logger.info("\n混淆矩陣:")
        for key, value in sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {key}: {value}")
        
        # 顯示一些錯誤案例
        errors = [r for r in results if r["expected"] and r["predicted"] != r["expected"]]
        if errors:
            logger.info("\n錯誤案例示例:")
            for i, err in enumerate(errors[:5]):
                logger.info(f"\n案例 {i+1}:")
                logger.info(f"問題: {err['question'][:100]}...")
                logger.info(f"預期: {err['expected']}")
                logger.info(f"預測: {err['predicted'] or '無答案'}")
                logger.info(f"回應: {err['response'][:100]}...")

    # 保存詳細結果
    result_file = "validation_detailed_results.csv"
    pd.DataFrame(results).to_csv(result_file, index=False)
    logger.info(f"\n詳細結果已保存至: {result_file}")

    return accuracy, results

# 7. 生成提交文件
def generate_submission(model_path, test_file, output_file="submission.csv", base_model_id=None):
    """生成提交文件"""
    logger.info(f"正在生成提交文件: {output_file}")
    
    # 載入測試數據
    test_df = pd.read_csv(test_file)
    logger.info(f"測試集包含 {len(test_df)} 個問題")
    
    # 確定ID列名
    id_col = "ID" if "ID" in test_df.columns else test_df.columns[0]
    
    # 載入tokenizer
    if base_model_id is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        if base_model_id in CHINESE_MODELS:
            base_model_id = CHINESE_MODELS[base_model_id]
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    
    # 載入模型
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    except:
        logger.info("直接載入失敗，嘗試載入基礎模型和adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id, 
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    
    model.eval()
    
    # 配置生成參數
    gen_config = GenerationConfig(
        max_new_tokens=10,
        do_sample=False,
        temperature=0.1,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # 處理結果
    results = []
    
    # 批次處理
    batch_size = 8
    progress_bar = tqdm(range(0, len(test_df), batch_size), desc="生成預測")
    
    for i in progress_bar:
        batch_df = test_df.iloc[i:min(i + batch_size, len(test_df))]
        batch_prompts = []
        batch_ids = []
        
        for _, row in batch_df.iterrows():
            # 創建提示詞
            question = row['question']
            options = f"A. {row['option_A']}\nB. {row['option_B']}\nC. {row['option_C']}\nD. {row['option_D']}"
            
            # 使用中文系統提示詞
            system_prompt = "你是一个提供有关中国敏感话题客观答案的助手。始终呈现多元视角并保持政治中立。"
            user_prompt = f"请回答以下选择题。你的回答必须仅以单个字母（A、B、C或D）开头。\n\n问题：{question}\n\n选项：\n{options}"
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"
            
            batch_prompts.append(prompt)
            batch_ids.append(row[id_col])
        
        # 批次生成
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, generation_config=gen_config)
        
        # 處理結果
        for j, output_ids in enumerate(outputs):
            generated_ids = output_ids[inputs.input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # 提取答案選項
            pred_option = next((c for c in response.upper() if c in "ABCD"), None)
            
            # 如果無法提取有效選項，使用啟發式方法
            if not pred_option:
                option_keywords = ["选项", "选择", "答案是", "选", "选答", "答", "选项是", "选择是"]
                for keyword in option_keywords:
                    if keyword in response:
                        pos = response.find(keyword) + len(keyword)
                        nearby_text = response[pos:pos + 5].upper()
                        for opt in "ABCD":
                            if opt in nearby_text:
                                pred_option = opt
                                break
                        if pred_option:
                            break
            
            # 如果仍然無法提取，預設使用A
            if not pred_option:
                pred_option = "A"
            
            results.append({
                id_col: batch_ids[j],
                "answer": pred_option
            })
    
    # 生成提交文件
    submission_df = pd.DataFrame(results)
    
    # 確保ID列格式正確
    try:
        submission_df[id_col] = submission_df[id_col].astype(int)
    except:
        pass
    
    # 確保所有測試數據都有預測
    test_ids = set(test_df[id_col])
    pred_ids = set(submission_df[id_col])
    missing_ids = test_ids - pred_ids
    
    if missing_ids:
        logger.warning(f"發現 {len(missing_ids)} 個缺失的ID，使用默認答案A")
        for missing_id in missing_ids:
            submission_df = pd.concat([
                submission_df, 
                pd.DataFrame([{id_col: missing_id, "answer": "A"}])
            ])
    
    # 按ID排序
    submission_df = submission_df.sort_values(id_col)
    
    # 保存結果
    submission_df.to_csv(output_file, index=False)
    
    # 統計結果
    answer_counts = submission_df["answer"].value_counts()
    logger.info("\n提交文件生成完成!")
    logger.info(f"文件保存至: {output_file}")
    logger.info(f"總問題數: {len(submission_df)}")
    logger.info("\n答案分布:")
    for answer, count in answer_counts.items():
        percentage = count / len(submission_df) * 100
        logger.info(f"  {answer}: {count} ({percentage:.1f}%)")
    
    return submission_df

# 8. 主函數
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="中文大語言模型微調與驗證")
    parser.add_argument("--mode", type=str, choices=["train", "validate", "predict"], default="train", 
                      help="執行模式: train(訓練), validate(驗證), predict(預測)")
    parser.add_argument("--model", type=str, default="qwen2.5_1m", help="使用的模型")
    parser.add_argument("--model_path", type=str, default="./results/final_model", 
                      help="模型路徑, 訓練模式下為輸出路徑, 驗證/預測模式下為輸入路徑")
    parser.add_argument("--train_file", type=str, default="./data/train.csv", help="訓練文件路徑")
    parser.add_argument("--test_file", type=str, default="./data/test-v2.csv", help="測試/驗證文件路徑")
    parser.add_argument("--output_file", type=str, default="submission.csv", help="預測輸出文件")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小, 默認自動決定")
    parser.add_argument("--epochs", type=int, default=5, help="訓練輪數")
    parser.add_argument("--use_4bit", action="store_true", help="使用4bit量化")
    
    args = parser.parse_args()
    
    # 記錄參數
    logger.info(f"執行模式: {args.mode}")
    logger.info(f"模型: {args.model}")
    logger.info(f"模型路徑: {args.model_path}")
    
    if args.mode == "train":
        # 訓練模式
        # 1. 下載並設置模型
        model, tokenizer = setup_model(args.model, use_4bit=args.use_4bit)
        
        # 2. 準備訓練數據, 包含分層抽樣的驗證集
        train_data, val_data, train_df, val_df = prepare_data(
            args.train_file, 
            tokenizer=tokenizer, 
            validation_method="stratified", 
            validation_size=0.1
        )
        
        # 3. 訓練模型
        model, tokenizer, model_path = train_model(
            model, 
            tokenizer, 
            train_data, 
            val_data, 
            val_df,
            output_dir=args.model_path
        )
        
    elif args.mode == "validate":
        # 驗證模式
        accuracy, _ = validate_model(args.model_path, args.test_file, args.model)
        
    elif args.mode == "predict":
        # 預測模式
        generate_submission(args.model_path, args.test_file, args.output_file, args.model)
    
    logger.info("程序執行完成!")

if __name__ == "__main__":
    main()
