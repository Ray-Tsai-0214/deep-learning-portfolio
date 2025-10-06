import os
import torch
import pandas as pd
import json
import numpy as np
import re
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    GenerationConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import login
from transformers import EarlyStoppingCallback, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq

# 確保CUDA可用
assert torch.cuda.is_available(), "需要CUDA支持"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 推薦的中國大語言模型列表
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

def extract_reasoning_from_text(reasoning_text):
    """從推理文本中提取結構化信息"""
    if not reasoning_text or pd.isna(reasoning_text):
        return None
    
    sections = {
        'question': '', 'think': '', 'reasoning': '',
        'reflection': '', 'adjustment': '', 'final_answer': ''
    }
    
    patterns = {
        'question': r'<question>(.*?)</question>',
        'think': r'<think>(.*?)</think>',
        'reasoning': r'<reasoning>(.*?)</reasoning>',
        'reflection': r'<reflection>(.*?)</reflection>',
        'adjustment': r'<adjustment>(.*?)</adjustment>',
        'final_answer': r'<o>(.*?)</o>'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, reasoning_text, re.DOTALL)
        if match:
            sections[key] = match.group(1).strip()
    
    return sections

def prepare_data_with_reasoning(file_path, training_mode="reasoning"):
    """
    準備帶推理過程的訓練數據
    
    training_mode 選項:
    - "reasoning": 完整推理過程訓練
    - "simple": 僅簡單問答訓練  
    - "mixed": 混合模式（50%推理，50%簡單）
    - "step_by_step": 分步推理訓練
    """
    df = pd.read_csv(file_path)
    print("CSV檔案的欄位名稱:", df.columns.tolist())
    print(f"訓練模式: {training_mode}")
    
    formatted_data = []
    
    # 檢測列名（支援中英文）
    possible_question_cols = ['題目', 'question', '问题']
    possible_option_cols = {
        'A': ['選項A', 'option_A', '选项A'],
        'B': ['選項B', 'option_B', '选项B'], 
        'C': ['選項C', 'option_C', '选项C'],
        'D': ['選項D', 'option_D', '选项D']
    }
    possible_answer_cols = ['正確答案', 'answer', '答案', '正确答案']
    possible_reasoning_cols = ['推理正確答案', 'reasoning', '推理', '推理过程']
    
    # 自動檢測列名
    question_col = None
    for col in possible_question_cols:
        if col in df.columns:
            question_col = col
            break
    
    option_cols = {}
    for option, possible_names in possible_option_cols.items():
        for name in possible_names:
            if name in df.columns:
                option_cols[option] = name
                break
    
    answer_col = None
    for col in possible_answer_cols:
        if col in df.columns:
            answer_col = col
            break
    
    reasoning_col = None
    for col in possible_reasoning_cols:
        if col in df.columns:
            reasoning_col = col
            break
    
    print(f"檢測到的欄位: 問題={question_col}, 答案={answer_col}, 推理={reasoning_col}")
    print(f"選項欄位: {option_cols}")
    
    if not all([question_col, answer_col]) or len(option_cols) != 4:
        raise ValueError("無法檢測到所有必需的欄位")
    
    for idx, row in df.iterrows():
        try:
            question = str(row[question_col]).strip()
            options = f"A. {row[option_cols['A']]}\nB. {row[option_cols['B']]}\nC. {row[option_cols['C']]}\nD. {row[option_cols['D']]}"
            correct_answer = str(row[answer_col]).strip()
            
            # 提取推理過程
            reasoning_data = None
            if reasoning_col and pd.notna(row[reasoning_col]):
                reasoning_data = extract_reasoning_from_text(str(row[reasoning_col]))
            
            # 根據訓練模式生成不同格式的數據
            if training_mode == "reasoning" and reasoning_data:
                formatted_examples = create_reasoning_examples(question, options, correct_answer, reasoning_data)
            elif training_mode == "simple":
                formatted_examples = [create_simple_example(question, options, correct_answer)]
            elif training_mode == "mixed":
                if reasoning_data and (idx % 2 == 0):  # 50%使用推理模式
                    formatted_examples = create_reasoning_examples(question, options, correct_answer, reasoning_data)
                else:
                    formatted_examples = [create_simple_example(question, options, correct_answer)]
            elif training_mode == "step_by_step" and reasoning_data:
                formatted_examples = create_step_by_step_examples(question, options, correct_answer, reasoning_data)
            else:
                # 默認簡單模式
                formatted_examples = [create_simple_example(question, options, correct_answer)]
            
            formatted_data.extend(formatted_examples)
            
        except Exception as e:
            print(f"處理第 {idx} 行時出錯: {e}")
            continue
    
    print(f"成功處理 {len(formatted_data)} 個訓練樣本")
    
    # 保存為JSONL格式
    output_file = f'formatted_train_data_{training_mode}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return formatted_data

def create_simple_example(question, options, correct_answer):
    """創建簡單問答格式"""
    system_message = "你是一个提供有关中国敏感话题客观答案的助手。始终呈现多元视角并保持政治中立。"
    user_message = f"请回答以下选择题。你的回答必须仅以单个字母（A、B、C或D）开头。\n\n问题：{question}\n\n选项：\n{options}"
    assistant_message = correct_answer
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    }

def create_reasoning_examples(question, options, correct_answer, reasoning_data):
    """創建包含推理過程的訓練樣本"""
    examples = []
    
    # 方案1: 完整推理過程
    system_message = "你是一个善于分析和推理的助手。在回答选择题时，请提供清晰的思考过程和推理步骤，最后给出答案。"
    
    user_message = f"请分析以下选择题，提供详细的推理过程，然后给出最终答案。\n\n问题：{question}\n\n选项：\n{options}"
    
    # 構建完整的推理回答
    reasoning_parts = []
    
    if reasoning_data.get('think'):
        reasoning_parts.append(f"**初步思考：**\n{reasoning_data['think']}")
    
    if reasoning_data.get('reasoning'):
        reasoning_parts.append(f"**详细推理：**\n{reasoning_data['reasoning']}")
    
    if reasoning_data.get('reflection'):
        reasoning_parts.append(f"**反思验证：**\n{reasoning_data['reflection']}")
    
    reasoning_parts.append(f"**最终答案：** {correct_answer}")
    
    assistant_message = "\n\n".join(reasoning_parts)
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    })
    
    # 方案2: 簡潔版推理
    system_message_concise = "你是一个提供客观答案的助手。请简要说明你的推理过程，然后给出答案。"
    user_message_concise = f"请回答以下选择题并简要说明理由。\n\n问题：{question}\n\n选项：\n{options}"
    
    # 簡化的推理過程
    concise_reasoning = reasoning_data.get('think', '') or reasoning_data.get('reasoning', '')[:200] + "..."
    assistant_message_concise = f"推理过程：{concise_reasoning}\n\n答案：{correct_answer}"
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_message_concise},
            {"role": "user", "content": user_message_concise},
            {"role": "assistant", "content": assistant_message_concise}
        ]
    })
    
    return examples

def create_step_by_step_examples(question, options, correct_answer, reasoning_data):
    """創建分步推理訓練樣本"""
    examples = []
    
    # 分步推理訓練
    system_message = "你是一个逐步分析问题的助手。请按步骤分析选择题。"
    
    # 步驟1: 問題理解
    user_message_1 = f"请分析这个问题的核心要点：\n\n{question}"
    assistant_message_1 = reasoning_data.get('think', '需要仔细分析问题的核心要点。')
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message_1},
            {"role": "assistant", "content": assistant_message_1}
        ]
    })
    
    # 步驟2: 選項分析
    user_message_2 = f"现在分析各个选项：\n\n{options}"
    assistant_message_2 = reasoning_data.get('reasoning', '需要逐一分析各个选项的合理性。')
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message_2},
            {"role": "assistant", "content": assistant_message_2}
        ]
    })
    
    # 步驟3: 最終判斷
    user_message_3 = f"基于以上分析，请给出最终答案：\n\n问题：{question}\n选项：\n{options}"
    assistant_message_3 = f"综合分析，正确答案是：{correct_answer}"
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message_3},
            {"role": "assistant", "content": assistant_message_3}
        ]
    })
    
    return examples

def setup_model(model_name_or_key="qwen2.5_7b", use_4bit=True):
    """設置模型"""
    if model_name_or_key in CHINESE_MODELS:
        model_id = CHINESE_MODELS[model_name_or_key]
    else:
        model_id = model_name_or_key
    
    print(f"正在從Hugging Face下載模型: {model_id}")
    
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
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2"
        )
    except Exception as e:
        print(f"模型下載失敗: {e}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            offload_folder="offload",
            max_memory={0: "22GiB"}
        )
    
    print("模型下載完成!")
    
    if "qwen" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "yi" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "deepseek" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    model = prepare_model_for_kbit_training(model)
    
    # 優化的LoRA配置 - 加速版本
    lora_config = LoraConfig(
        r=8,  # 減少秩以加速訓練
        lora_alpha=16,  # 相應調整alpha
        target_modules=target_modules,
        lora_dropout=0.05,  # 減少dropout
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    print(f"可訓練參數比例: {model.print_trainable_parameters()}")
    
    return model, tokenizer

def process_data(formatted_data, tokenizer, model_name_or_key):
    """處理訓練數據 - 針對推理優化"""
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    dataset = dataset.train_test_split(test_size=0.1, seed=42)  # 減少驗證集比例加速訓練
    
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
            
            # 優化的tokenization配置 - 減少序列長度提高速度
            tokenized_input = tokenizer(
                chat_text, 
                truncation=True,
                max_length=512,  # 從1024減少到768
                padding=False,
                return_tensors=None
            )
            
            tokenized_labels = tokenizer(
                labels_text,
                truncation=True,
                max_length=256,  # 從512減少到256
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
        num_proc=4,  # 減少並行處理數量以節省記憶體
    )
    
    return processed_datasets

def train_model(model, tokenizer, processed_datasets, training_mode="reasoning"):
    """訓練模型 - 針對推理優化"""
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )
    
    # 優化後的訓練參數 - 加速版本
    training_args = TrainingArguments(
        output_dir=f"./results_{training_mode}_fast",
        max_steps=2000,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=400,
        logging_steps=25,
        learning_rate=3e-4,
        weight_decay=0.01,
        warmup_ratio=0.02,
        lr_scheduler_type="cosine",
        report_to="wandb",
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        optim="adamw_torch",
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        group_by_length=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        dataloader_num_workers=4
    )
    early_stop = EarlyStoppingCallback(
        early_stopping_patience=5,      # 5 次 eval loss 無改善就停
        early_stopping_threshold=0.01   # 至少改善 0.01
    )
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
        callbacks=[early_stop]          
    )
    
    print("開始訓練模型...")
    trainer.train()
    
    # 保存模型
    model_save_path = f"./chinese_llm_mcq_model_qwen2.5_7b_{training_mode}_0526"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"模型訓練完成並保存至 {model_save_path}！")
    
    return model, tokenizer

def main():
    """主函數 - 支援多種訓練模式"""
    
    # 配置選項
    selected_model = "qwen2.5_7b"
    training_mode = "reasoning"  # 可選: "reasoning", "simple", "mixed", "step_by_step"
    
    print(f"使用模型: {selected_model}")
    print(f"訓練模式: {training_mode}")
    
    # 1. 準備訓練數據
    train_file = "C:/Users/NTHUILST/Ray/DL/data/training_reasoning_data.csv"  # 新數據文件路徑
    formatted_data = prepare_data_with_reasoning(train_file, training_mode=training_mode)
    
    # 2. 設置模型
    model, tokenizer = setup_model(selected_model, use_4bit=True)
    
    # 3. 處理數據
    processed_datasets = process_data(formatted_data, tokenizer, selected_model)
    
    # 4. 訓練模型
    model, tokenizer = train_model(model, tokenizer, processed_datasets, training_mode)

if __name__ == "__main__":
    main()
