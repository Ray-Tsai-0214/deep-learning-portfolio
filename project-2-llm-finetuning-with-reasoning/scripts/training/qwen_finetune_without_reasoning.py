import os
import torch
import pandas as pd
import json
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import login

# 確保CUDA可用
assert torch.cuda.is_available(), "需要CUDA支持"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 推薦的中國大語言模型列表
CHINESE_MODELS = {
    "qwen2_7b": "Qwen/Qwen2-7B",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5_14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5_1m": "Qwen/Qwen2.5-14B-Instruct-1M",
    "qwen1.5_7b": "Qwen/Qwen1.5-7B",
    "qwen1.5_14b": "Qwen/Qwen1.5-14B",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek_qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "yi_6b": "01-ai/Yi-6B"
}

def prepare_simple_qa_data(file_path):
    """
    準備簡單問答數據（無推理過程）
    適用於純粹的選擇題任務
    """
    df = pd.read_csv(file_path)
    print(f"數據集大小: {len(df)} 行")
    print("CSV檔案的欄位名稱:", df.columns.tolist())
    
    formatted_data = []
    
    # 檢測列名
    question_col = '題目'
    option_cols = {
        'A': '選項A', 'B': '選項B', 'C': '選項C', 'D': '選項D'
    }
    answer_col = '正確答案'
    
    print(f"檢測到的欄位: 問題={question_col}, 答案={answer_col}")
    print(f"選項欄位: {option_cols}")
    
    for idx, row in df.iterrows():
        try:
            question = str(row[question_col]).strip()
            options = f"A. {row[option_cols['A']]}\nB. {row[option_cols['B']]}\nC. {row[option_cols['C']]}\nD. {row[option_cols['D']]}"
            correct_answer = str(row[answer_col]).strip()
            
            # 創建多種格式的訓練樣本提高泛化能力
            examples = create_qa_examples(question, options, correct_answer)
            formatted_data.extend(examples)
            
            if (idx + 1) % 5000 == 0:
                print(f"已處理 {idx + 1} 行數據...")
                
        except Exception as e:
            print(f"處理第 {idx} 行時出錯: {e}")
            continue
    
    print(f"成功處理 {len(formatted_data)} 個訓練樣本")
    return formatted_data

def create_qa_examples(question, options, correct_answer):
    """創建多種格式的問答樣本"""
    examples = []
    
    # 格式1: 標準問答格式
    system_message = "你是一個專業的選擇題解答助手。請仔細閱讀題目和選項，給出正確答案。"
    user_message = f"請回答以下選擇題，只需要給出答案字母(A、B、C或D)：\n\n題目：{question}\n\n選項：\n{options}"
    assistant_message = correct_answer
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    })
    
    # 格式2: 簡潔格式
    system_message_2 = "你是一個選擇題解答專家。"
    user_message_2 = f"選擇題：{question}\n\n{options}\n\n答案是："
    assistant_message_2 = correct_answer
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_message_2},
            {"role": "user", "content": user_message_2},
            {"role": "assistant", "content": assistant_message_2}
        ]
    })
    
    # 格式3: 直接格式（用於提高回答準確性）
    system_message_3 = "根據題目內容選擇正確答案。"
    user_message_3 = f"{question}\n{options}\n正確答案："
    assistant_message_3 = correct_answer
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_message_3},
            {"role": "user", "content": user_message_3},
            {"role": "assistant", "content": assistant_message_3}
        ]
    })
    
    return examples

def setup_model(model_name_or_key="qwen2.5_14b", use_4bit=True):
    """設置模型 - 針對RTX 4090優化"""
    if model_name_or_key in CHINESE_MODELS:
        model_id = CHINESE_MODELS[model_name_or_key]
    else:
        model_id = model_name_or_key
    
    print(f"正在從Hugging Face下載模型: {model_id}")
    
    # RTX 4090 24GB配置 - 可以使用更激進的設置
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  # 使用bfloat16提高性能
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        quantization_config = None
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"Flash Attention加載失敗，使用標準配置: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    
    print("模型下載完成!")
    
    # 設置LoRA適配器
    if "qwen" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "yi" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "deepseek" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    model = prepare_model_for_kbit_training(model)
    
    # 簡單問答任務的LoRA配置
    lora_config = LoraConfig(
        r=16,  # 適中的秩，平衡性能和效果
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    print(f"可訓練參數比例: {model.print_trainable_parameters()}")
    
    return model, tokenizer

def process_data(formatted_data, tokenizer):
    """處理簡單問答數據 - RTX 4090優化版本"""
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    dataset = dataset.train_test_split(test_size=0.05, seed=42)  # 減少驗證集比例
    
    def preprocess_function(examples):
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for messages in examples["messages"]:
            chat_text = ""
            labels_text = ""
            
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    chat_text += f"<|im_start|>system\n{message['content']}<|im_end|>\n"
                elif message["role"] == "user":
                    chat_text += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                    chat_text += "<|im_start|>assistant\n"
                elif message["role"] == "assistant" and i > 0:
                    labels_text = f"{message['content']}<|im_end|>"
            
            # 簡單問答任務 - 使用較短的序列長度
            tokenized_input = tokenizer(
                chat_text, 
                truncation=True,
                max_length=512,  # 簡單問答不需要太長的上下文
                padding=False,
                return_tensors=None
            )
            
            tokenized_labels = tokenizer(
                labels_text,
                truncation=True,
                max_length=16,  # 答案通常很短
                padding=False,
                return_tensors=None
            )
            
            input_ids = tokenized_input["input_ids"]
            combined_input_ids = input_ids + tokenized_labels["input_ids"]
            attention_mask = [1] * len(combined_input_ids)
            labels = [-100] * len(input_ids) + tokenized_labels["input_ids"]
            
            all_input_ids.append(combined_input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels
        }
    
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4,
    )
    
    return processed_datasets

def train_model(model, tokenizer, processed_datasets, model_name):
    """訓練模型 - RTX 4090 24GB優化配置"""
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )
    
    # RTX 4090 24GB 專用配置 - 充分利用大顯存
    training_args = TrainingArguments(
        output_dir=f"./results_simple_qa_{model_name}",
        max_steps=1500,  # 簡單任務步數可以更少
        per_device_train_batch_size=8,  # 24GB顯存可以用更大batch size
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # 有效batch size = 8*4 = 32
        evaluation_strategy="steps",
        eval_steps=150,  # 每150步評估一次
        save_strategy="steps",
        save_steps=300,
        logging_steps=20,
        learning_rate=3e-4,  # 簡單任務可以用更高學習率
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        report_to="wandb",
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,  # RTX 4090支持bfloat16
        optim="adamw_torch",
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        group_by_length=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        dataloader_num_workers=6,  # 增加數據加載線程
        # 24GB顯存專用設置
        dataloader_pin_memory=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    
    # 早停機制
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )
    
    print("開始訓練簡單問答模型...")
    print(f"訓練樣本數: {len(processed_datasets['train'])}")
    print(f"驗證樣本數: {len(processed_datasets['test'])}")
    print(f"有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    trainer.train()
    
    # 保存模型
    model_save_path = f"./chinese_llm_simple_qa_{model_name}"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"模型訓練完成並保存至 {model_save_path}！")
    
    return model, tokenizer

def main():
    """主函數 - 簡單問答訓練"""
    
    # 配置選項
    selected_model = "qwen2.5_14b"  # 推薦14B模型平衡效果和速度
    print(f"使用模型: {selected_model}")
    print("訓練模式: 簡單問答（無推理過程）")
    
    # 1. 準備訓練數據
    train_file = "C:/Users/NTHUILST/Ray/DL/data/training_data_without_reasoning.csv"
    formatted_data = prepare_simple_qa_data(train_file)
    
    # 2. 設置模型
    model, tokenizer = setup_model(selected_model, use_4bit=True)
    
    # 3. 處理數據
    processed_datasets = process_data(formatted_data, tokenizer)
    
    # 4. 訓練模型
    model, tokenizer = train_model(model, tokenizer, processed_datasets, selected_model)

if __name__ == "__main__":
    main()
