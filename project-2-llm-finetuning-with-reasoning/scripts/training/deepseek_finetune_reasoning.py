import os
import torch
import pandas as pd
import json
import numpy as np
import re
import logging
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

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 確保CUDA可用
assert torch.cuda.is_available(), "需要CUDA支持"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# DeepSeek模型配置
DEEPSEEK_MODELS = {
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek_chat_7b": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek_r1_7b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek_r1_14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek_r1_32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek_coder_7b": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "deepseek_math_7b": "deepseek-ai/deepseek-math-7b-instruct"
}

def read_tsv_data(file_path):
    """專門讀取TSV格式的訓練數據"""
    logger.info(f"讀取TSV文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        # 使用tab分隔符讀取TSV文件
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8-sig')
        logger.info(f"成功讀取 {len(df)} 行數據")
        logger.info(f"欄位名稱: {list(df.columns)}")
        
        # 基本數據驗證
        required_cols = ['題目', '選項A', '選項B', '選項C', '選項D', '正確答案']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"缺少必要欄位: {missing_cols}")
            # 嘗試映射列名
            df = map_column_names(df)
        
        # 檢查推理欄位
        reasoning_cols = ['推理正確答案', '推理', 'reasoning', '推理过程']
        reasoning_col = None
        for col in reasoning_cols:
            if col in df.columns:
                reasoning_col = col
                break
        
        if reasoning_col:
            reasoning_count = df[reasoning_col].notna().sum()
            logger.info(f"發現推理欄位 '{reasoning_col}': {reasoning_count}/{len(df)} 行有推理數據")
        else:
            logger.warning("未找到推理欄位")
        
        # 答案分佈檢查
        if '正確答案' in df.columns:
            answer_dist = df['正確答案'].value_counts()
            logger.info(f"答案分佈: {answer_dist.to_dict()}")
        
        return df
        
    except Exception as e:
        logger.error(f"讀取TSV文件失敗: {e}")
        raise
def map_column_names(df):
    """映射可能的列名變體"""
    column_mapping = {
        'Question': '題目',
        'question': '題目',
        'Option A': '選項A',
        'Option B': '選項B', 
        'Option C': '選項C',
        'Option D': '選項D',
        'Answer': '正確答案',
        'answer': '正確答案',
        'Reasoning': '推理正確答案',
        'reasoning': '推理正確答案'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
            logger.info(f"映射列名: {old_name} → {new_name}")
    
    return df

def extract_reasoning_enhanced(reasoning_text):
    """增強版推理文本提取，處理TSV格式的推理數據"""
    if not reasoning_text or pd.isna(reasoning_text):
        return None
    
    # 清理和預處理文本
    reasoning_text = str(reasoning_text).strip()
    
    # 如果文本太短，直接返回
    if len(reasoning_text) < 10:
        return None
    
    sections = {
        'question': '', 'think': '', 'reasoning': '',
        'reflection': '', 'adjustment': '', 'final_answer': ''
    }
    
    # 多種推理格式的模式匹配
    patterns = {
        'question': [
            r'<question>(.*?)</question>',
            r'問題[：:](.*?)(?=\n|思考|推理|$)',
            r'Question[：:]?(.*?)(?=\n|Think|Reasoning|$)'
        ],
        'think': [
            r'<think>(.*?)</think>',
            r'思考[：:](.*?)(?=\n|推理|分析|$)',
            r'Think[：:]?(.*?)(?=\n|Reasoning|Analysis|$)',
            r'初步思考[：:](.*?)(?=\n|詳細|推理|$)'
        ],
        'reasoning': [
            r'<reasoning>(.*?)</reasoning>',
            r'推理[：:](.*?)(?=\n|反思|調整|答案|$)',
            r'Reasoning[：:]?(.*?)(?=\n|Reflection|Answer|$)',
            r'詳細推理[：:](.*?)(?=\n|反思|最終|$)',
            r'step \d+[：:]?(.*?)(?=step|\n|$)'
        ],
        'reflection': [
            r'<reflection>(.*?)</reflection>',
            r'反思[：:](.*?)(?=\n|調整|答案|$)',
            r'Reflection[：:]?(.*?)(?=\n|Adjustment|Answer|$)',
            r'驗證[：:](.*?)(?=\n|答案|$)'
        ],
        'adjustment': [
            r'<adjustment>(.*?)</adjustment>',
            r'調整[：:](.*?)(?=\n|答案|$)',
            r'Adjustment[：:]?(.*?)(?=\n|Answer|$)'
        ],
        'final_answer': [
            r'<o>(.*?)</o>',
            r'答案[：:](.*?)(?=\n|$)',
            r'Answer[：:]?(.*?)(?=\n|$)',
            r'最終答案[：:](.*?)(?=\n|$)',
            r'正確答案[：:](.*?)(?=\n|$)'
        ]
    }
    
    # 提取各個部分
    for key, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, reasoning_text, re.DOTALL | re.IGNORECASE)
            if matches:
                # 合併所有匹配的內容並清理
                combined_text = ' '.join(match.strip() for match in matches)
                sections[key] = clean_text(combined_text)
                break
    
    # 如果沒有找到結構化內容，嘗試智能分割
    if not any(sections.values()):
        sections = smart_split_reasoning(reasoning_text)
    
    return sections

def clean_text(text):
    """清理文本內容"""
    if not text:
        return ""
    
    # 移除多餘的空白和特殊字符
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('"', "'")
    text = text.strip()
    
    # 限制長度
    if len(text) > 300:
        text = text[:300] + "..."
    
    return text

def smart_split_reasoning(text):
    """智能分割推理文本"""
    sections = {
        'question': '', 'think': '', 'reasoning': '',
        'reflection': '', 'adjustment': '', 'final_answer': ''
    }
    
    # 簡單的文本分割策略
    sentences = text.split('。')
    if len(sentences) >= 3:
        sections['think'] = sentences[0] + "。"
        sections['reasoning'] = "。".join(sentences[1:-1]) + "。"
        sections['final_answer'] = sentences[-1]
    else:
        sections['reasoning'] = text
    
    return sections
def prepare_reasoning_data_tsv(file_path, training_mode="mixed", max_samples=None, data_balance=True):
    """專門處理TSV格式的推理數據準備"""
    
    # 讀取TSV數據
    df = read_tsv_data(file_path)
    
    # 數據清理
    logger.info("開始數據清理...")
    
    # 移除空值
    initial_count = len(df)
    df = df.dropna(subset=['題目', '正確答案'])
    logger.info(f"移除空值: {initial_count} → {len(df)} 行")
    
    # 驗證答案格式
    valid_answers = {'A', 'B', 'C', 'D'}
    df = df[df['正確答案'].isin(valid_answers)]
    logger.info(f"驗證答案格式後: {len(df)} 行")
    
    # 去重
    df = df.drop_duplicates(subset=['題目'])
    logger.info(f"去重後: {len(df)} 行")
    
    # 數據平衡
    if data_balance:
        min_count = df['正確答案'].value_counts().min()
        balanced_dfs = []
        for answer in ['A', 'B', 'C', 'D']:
            answer_df = df[df['正確答案'] == answer]
            if len(answer_df) > min_count:
                answer_df = answer_df.sample(n=min_count, random_state=42)
            balanced_dfs.append(answer_df)
        df = pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)
        logger.info(f"數據平衡後: {len(df)} 行")
    
    # 限制樣本數量
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        logger.info(f"限制樣本數量至: {len(df)} 行")
    
    # 檢查推理欄位
    reasoning_cols = ['推理正確答案', '推理', 'reasoning', '推理过程']
    reasoning_col = None
    for col in reasoning_cols:
        if col in df.columns:
            reasoning_col = col
            break
    
    if not reasoning_col:
        logger.warning("未找到推理欄位，將使用簡單問答模式")
        training_mode = "simple"
    
    logger.info(f"使用推理欄位: {reasoning_col}")
    logger.info(f"訓練模式: {training_mode}")
    
    # 生成訓練樣本
    formatted_data = []
    
    for idx, row in df.iterrows():
        try:
            question = str(row['題目']).strip()
            options = f"A. {row['選項A']}\nB. {row['選項B']}\nC. {row['選項C']}\nD. {row['選項D']}"
            correct_answer = str(row['正確答案']).strip()
            
            # 提取推理過程
            reasoning_data = None
            if reasoning_col and pd.notna(row[reasoning_col]):
                reasoning_data = extract_reasoning_enhanced(str(row[reasoning_col]))
            
            # 根據訓練模式生成數據
            examples = []
            
            if training_mode == "reasoning" and reasoning_data:
                examples = create_deepseek_reasoning_examples(question, options, correct_answer, reasoning_data)
            elif training_mode == "mixed":
                # 總是包含簡單樣本
                examples.append(create_deepseek_simple_example(question, options, correct_answer))
                # 如果有推理數據，添加推理樣本
                if reasoning_data:
                    examples.extend(create_deepseek_reasoning_examples(question, options, correct_answer, reasoning_data))
            elif training_mode == "step_by_step" and reasoning_data:
                examples = create_deepseek_step_examples(question, options, correct_answer, reasoning_data)
            else:
                # 默認簡單模式
                examples = [create_deepseek_simple_example(question, options, correct_answer)]
            
            formatted_data.extend(examples)
            
        except Exception as e:
            logger.warning(f"處理第 {idx} 行時出錯: {e}")
            continue
    
    logger.info(f"成功生成 {len(formatted_data)} 個訓練樣本")
    
    # 保存為JSONL格式
    output_file = f'deepseek_train_data_{training_mode}.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"訓練數據已保存至: {output_file}")
    
    return formatted_data
def create_deepseek_simple_example(question, options, correct_answer):
    """為DeepSeek創建簡單問答樣本"""
    system_message = "You are a helpful assistant that provides objective answers to multiple choice questions. Always respond with the correct letter only."
    
    user_message = f"請回答以下選擇題，只需回答選項字母（A、B、C或D）：\n\n問題：{question}\n\n選項：\n{options}\n\n答案："
    
    assistant_message = correct_answer
    
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    }

def create_deepseek_reasoning_examples(question, options, correct_answer, reasoning_data):
    """為DeepSeek創建推理樣本"""
    examples = []
    
    # 樣本1: 簡潔推理模式
    system_message = "You are an analytical assistant. Provide brief reasoning before giving your answer to multiple choice questions."
    
    user_message = f"請分析以下選擇題並簡要說明推理過程，然後給出答案：\n\n問題：{question}\n\n選項：\n{options}"
    
    # 構建推理回答
    reasoning_parts = []
    
    if reasoning_data.get('think'):
        reasoning_parts.append(f"思考：{reasoning_data['think'][:150]}")
    
    if reasoning_data.get('reasoning'):
        reasoning_parts.append(f"推理：{reasoning_data['reasoning'][:200]}")
    
    reasoning_parts.append(f"答案：{correct_answer}")
    
    assistant_message = "\n\n".join(reasoning_parts)
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    })
    
    return examples

def create_deepseek_step_examples(question, options, correct_answer, reasoning_data):
    """為DeepSeek創建分步推理樣本"""
    examples = []
    
    system_message = "You are a systematic problem-solving assistant. Analyze multiple choice questions step by step."
    
    user_message = f"請按步驟分析以下選擇題：\n\n問題：{question}\n\n選項：\n{options}"
    
    # 構建分步回答
    steps = []
    
    if reasoning_data.get('think'):
        steps.append(f"步驟1 - 理解問題：{reasoning_data['think'][:100]}")
    
    if reasoning_data.get('reasoning'):
        steps.append(f"步驟2 - 分析選項：{reasoning_data['reasoning'][:150]}")
    
    if reasoning_data.get('reflection'):
        steps.append(f"步驟3 - 驗證答案：{reasoning_data['reflection'][:100]}")
    
    steps.append(f"最終答案：{correct_answer}")
    
    assistant_message = "\n\n".join(steps)
    
    examples.append({
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
    })
    
    return examples
def setup_deepseek_model(model_key="deepseek_r1_14b", use_4bit=True):
    """設置DeepSeek模型 - 針對推理任務優化"""
    
    if model_key in DEEPSEEK_MODELS:
        model_id = DEEPSEEK_MODELS[model_key]
    else:
        model_id = model_key
    
    logger.info(f"正在加載DeepSeek模型: {model_id}")
    
    # DeepSeek專用量化配置
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["lm_head"]
        )
    else:
        quantization_config = None
    
    # 加載tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        padding_side="left",
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加載模型
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        logger.info("使用Flash Attention 2")
    except Exception as e:
        logger.warning(f"Flash Attention 2不可用: {e}")
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    logger.info("DeepSeek模型加載完成!")
    
    # DeepSeek專用LoRA配置
    if "32b" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_r, lora_alpha = 16, 32
    elif "14b" in model_id.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_r, lora_alpha = 12, 24
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        lora_r, lora_alpha = 8, 16
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False
    )
    
    model = get_peft_model(model, lora_config)
    logger.info(f"LoRA配置完成: {model.print_trainable_parameters()}")
    
    return model, tokenizer
def process_deepseek_data(formatted_data, tokenizer, max_length=1024):
    """處理DeepSeek訓練數據"""
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    
    # 根據數據集大小調整分割比例
    test_size = 0.05 if len(dataset) > 10000 else 0.1
    dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    def preprocess_function(examples):
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for messages in examples["messages"]:
            # DeepSeek對話格式
            chat_text = ""
            labels_text = ""
            
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    chat_text += f"System: {message['content']}\n\n"
                elif message["role"] == "user":
                    chat_text += f"User: {message['content']}\n\n"
                    chat_text += "Assistant: "
                elif message["role"] == "assistant":
                    labels_text = message['content']
            
            # Tokenization
            tokenized_input = tokenizer(
                chat_text, 
                truncation=True,
                max_length=max_length - 200,
                padding=False,
                return_tensors=None
            )
            
            tokenized_labels = tokenizer(
                labels_text,
                truncation=True,
                max_length=200,
                padding=False,
                return_tensors=None
            )
            
            input_ids = tokenized_input["input_ids"]
            combined_input_ids = input_ids + tokenized_labels["input_ids"]
            attention_mask = [1] * len(combined_input_ids)
            labels = [-100] * len(input_ids) + tokenized_labels["input_ids"]
            
            # 長度檢查
            if len(combined_input_ids) > max_length:
                combined_input_ids = combined_input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                labels = labels[:max_length]
            
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
        num_proc=2,
    )
    
    return processed_datasets
def train_deepseek_reasoning(model, tokenizer, processed_datasets, training_mode, model_key):
    """訓練DeepSeek推理模型"""
    
    dataset_size = len(processed_datasets["train"])
    logger.info(f"訓練集大小: {dataset_size}")
    
    # 根據數據集大小調整參數
    if dataset_size > 50000:
        max_steps = 4000
        eval_steps = 400
        save_steps = 800
        batch_size = 4
        grad_accum = 16
    elif dataset_size > 20000:
        max_steps = 3000
        eval_steps = 300
        save_steps = 600
        batch_size = 6
        grad_accum = 12
    else:
        max_steps = 2000
        eval_steps = 200
        save_steps = 400
        batch_size = 8
        grad_accum = 8
    
    # 訓練參數
    training_args = TrainingArguments(
        output_dir=f"./deepseek_reasoning_{model_key}_{training_mode}",
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=grad_accum,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        logging_steps=50,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        optim="adamw_torch_fused",
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        group_by_length=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        report_to=None
    )
    
    # 早停回調
    early_stop = EarlyStoppingCallback(
        early_stopping_patience=8,
        early_stopping_threshold=0.005
    )
    
    # 數據整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stop]
    )
    
    logger.info("開始訓練DeepSeek推理模型...")
    trainer.train()
    
    # 保存最終模型
    final_model_path = f"./deepseek_reasoning_final_{model_key}_{training_mode}"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info(f"模型訓練完成並保存至: {final_model_path}")
    
    return model, tokenizer
def main():
    """主函數 - TSV推理訓練版本"""
    
    # 🎯 專為TSV推理訓練的配置
    config = {
        "model": "deepseek_r1_14b",        # DeepSeek R1 14B模型，適合推理任務
        "training_mode": "mixed",          # 混合模式：結合簡單問答和推理
        "max_samples": 15000,              # 控制訓練規模，避免過度訓練
        "data_balance": True,              # 平衡答案分佈
        "use_4bit": True                   # 使用4bit量化節省顯存
    }
    
    logger.info("=" * 70)
    logger.info("🧠 DeepSeek TSV推理訓練腳本")
    logger.info("=" * 70)
    logger.info(f"配置參數: {config}")
    
    # 🔥 使用您下載的TSV文件
    tsv_file = "C:/Users/NTHUILST/Ray/DL/data/training_data_fixed.tsv"
    
    if not os.path.exists(tsv_file):
        logger.error(f"TSV文件不存在: {tsv_file}")
        logger.info("請確保已從Google試算表下載TSV格式文件")
        return
    
    try:
        # 1. 準備TSV推理數據
        logger.info("📊 準備TSV推理訓練數據...")
        formatted_data = prepare_reasoning_data_tsv(
            tsv_file,
            training_mode=config["training_mode"],
            max_samples=config["max_samples"],
            data_balance=config["data_balance"]
        )
        
        # 2. 設置DeepSeek模型
        logger.info("🤖 設置DeepSeek模型...")
        model, tokenizer = setup_deepseek_model(
            config["model"],
            use_4bit=config["use_4bit"]
        )
        
        # 3. 處理訓練數據
        logger.info("⚙️  處理訓練數據...")
        processed_datasets = process_deepseek_data(formatted_data, tokenizer)
        
        # 4. 開始訓練
        logger.info("🚀 開始DeepSeek推理訓練...")
        model, tokenizer = train_deepseek_reasoning(
            model, tokenizer, processed_datasets,
            config["training_mode"], config["model"]
        )
        
        logger.info("=" * 70)
        logger.info("🎉 DeepSeek推理模型訓練完成!")
        logger.info("📈 預期Kaggle分數提升至: 0.75-0.85")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"訓練過程中出現錯誤: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()