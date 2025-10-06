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
    GenerationConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from huggingface_hub import login

# 设置Hugging Face Token (如果需要访问私有模型或加速下载)
# login(token="your_huggingface_token")  # 如需使用，请取消注释并填入你的token

# 确保CUDA可用
assert torch.cuda.is_available(), "需要CUDA支持"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 推荐的中国大语言模型列表
CHINESE_MODELS = {
    "qwen2_7b": "Qwen/Qwen2-7B",
    "qwen_QwQ": "Qwen/QwQ-32B",
    "qwen1.5_7b": "Qwen/Qwen1.5-7B",
    "qwen2.5_1m" :"Qwen/Qwen2.5-14B-Instruct-1M",
    "qwen1.5_14b": "Qwen/Qwen1.5-14B",
    "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek_qwen-14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek_qwen-32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "deepseek_lamma":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
}

# 1. 准备数据
def prepare_data(file_path):
    """從CSV文件準備訓練數據，轉換為JSONL格式"""
    df = pd.read_csv(file_path)
    print("CSV檔案的欄位名稱:", df.columns.tolist())
    
    formatted_data = []
    
    # 替換下面的欄位名稱為你的CSV實際的欄位名稱
    
    question_col = 'question'  # 或者你的CSV中實際的問題欄位名稱
    option_a_col = 'option_A'  # 或者選項A的實際欄位名稱
    option_b_col = 'option_B'  # 或者選項B的實際欄位名稱
    option_c_col = 'option_C'  # 或者選項C的實際欄位名稱
    option_d_col = 'option_D'  # 或者選項D的實際欄位名稱
    answer_col = 'answer'  # 或者答案的實際欄位名稱
    for _, row in df.iterrows():
        question = row[question_col]
        options = f"A. {row[option_a_col]}\nB. {row[option_b_col]}\nC. {row[option_c_col]}\nD. {row[option_d_col]}"
        
        # 构建Qwen格式的系统和用户指令
        system_message = "You are a helpful assistant that provides objective answers to questions about sensitive topics related to China. Always present diverse perspectives and maintain political neutrality."
        
        user_message = f"Please answer the following multiple-choice question. You MUST start your answer with a single letter (A, B, C, or D).\n\nQuestion: {question}\n\nOptions:\n{options}"
        
        # 构建助手回复
        assistant_message = row[answer_col]
        
        # 使用Qwen的格式整合
        formatted_example = {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
        }
        
        formatted_data.append(formatted_example)
    # 保存為JSONL格式用於訓練
    with open('formatted_train_data.jsonl', 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return formatted_data

# 2. 从Hugging Face下载并设置模型
def setup_model(model_name_or_key="qwen2.5_1m", use_4bit=True):
    """
    从Hugging Face下载并设置模型
    
    参数:
    - model_name_or_key: 模型名称（从CHINESE_MODELS字典中选择）或直接使用Hugging Face模型ID
    - use_4bit: 是否使用4-bit量化（推荐用于RTX 4090）
    """
    # 获取模型ID
    if model_name_or_key in CHINESE_MODELS:
        model_id = CHINESE_MODELS[model_name_or_key]
    else:
        model_id = model_name_or_key  # 直接使用作为模型ID
    
    print(f"正在从Hugging Face下载模型: {model_id}")
    
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
    
    # 下载并加载tokenizer
    print("下载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        padding_side="left"
    )
    
    # 确保tokenizer有padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 下载并加载模型
    print("下载模型（这可能需要几分钟时间）...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # 使用半精度加载
            attn_implementation="flash_attention_2"  # 啟用FlashAttention 2
        )
    except Exception as e:
        print(f"模型下载失败: {e}")
        print("尝试使用较大超时时间重新下载...")
        model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder="offload",  # 如果RAM不足可以使用磁盘卸载
        max_memory={0: "22GiB"}  # 限制GPU内存使用
    )
    
    print("模型下载完成!")
    
    # 根据模型类型确定target_modules
    # 不同模型的结构可能不同，需要适配不同的target_modules
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
        # 默认设置适用于大多数模型
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # 为kbit训练准备模型
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        r=16,  # 适中的LoRA秩，平衡性能和显存
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用LoRA配置
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    print(f"可训练参数比例: {model.print_trainable_parameters()}")
    
    return model, tokenizer

# 3. 数据处理函数
def process_data(formatted_data, tokenizer, model_name_or_key):
    """处理和准备训练数据"""
    # 创建Dataset对象
    dataset = Dataset.from_pandas(pd.DataFrame(formatted_data))
    
    # 训练/验证集分割
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # 定义预处理函数
    def preprocess_function(examples):
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for messages in examples["messages"]:
            chat_text = ""
            labels_text = ""
            
            # 构建完整对话文本
            for i, message in enumerate(messages):
                if message["role"] == "system":
                    chat_text += f"<|im_start|>system\n{message['content']}<|im_end|>\n"
                elif message["role"] == "user":
                    chat_text += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                    # 用户消息后的助手前缀，但不包含助手内容
                    chat_text += "<|im_start|>assistant\n"
                elif message["role"] == "assistant" and i > 0:  # 确保这是用户消息后的助手回复
                    # 助手内容用于标签
                    labels_text = f"{message['content']}<|im_end|>"
            
            # 编码输入部分
            tokenized_input = tokenizer(
                chat_text, 
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors=None
            )
            
            # 编码输出标签部分
            tokenized_labels = tokenizer(
                labels_text,
                truncation=True,
                max_length=48,
                padding=False,
                return_tensors=None
            )
            
            # 组合输入和标签
            input_ids = tokenized_input["input_ids"]
            combined_input_ids = input_ids + tokenized_labels["input_ids"]
            attention_mask = [1] * len(combined_input_ids)
            
            # 标签: 输入部分为-100，输出部分为实际token ID
            labels = [-100] * len(input_ids) + tokenized_labels["input_ids"]
            
            all_input_ids.append(combined_input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)
        
        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels
        }
    
    # 应用预处理
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,
    )
    
    return processed_datasets

# 4. 训练模型函数
def train_model(model, tokenizer, processed_datasets):
    """训练模型，针对RTX 4090优化"""
    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,  # 对于混合精度训练很重要
    )
    
    # 为4090 GPU优化的训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # 基于4090的24GB VRAM优化
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # 累积梯度以模拟更大批量
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=20,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb",  # 使用wandb记录训练过程
        gradient_checkpointing=True,  # 启用梯度检查点以节省内存
        fp16=False,  # 使用混合精度训练
        bf16=True,  # 如果你的GPU支持bf16，可以改为True
        optim="adamw_torch",
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
        group_by_length=True,  # 可加速训练
        load_best_model_at_end=True
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets["train"],
        eval_dataset=processed_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("开始训练模型...")
    trainer.train()
    
    # 保存模型
    model.save_pretrained("./chinese_llm_mcq_model_qwen2.5_1m")
    tokenizer.save_pretrained("./chinese_llm_mcq_model_qwen2.5_1m")
    print("模型训练完成并保存！")
    
    return model, tokenizer

# 主函数
def main():
    # 选择要使用的模型
    selected_model = "qwen2.5_1m"  # 可以从CHINESE_MODELS中选择
    
    # 1. 准备训练数据
    train_file = "C:/Users/NTHUILST/Ray/DL/data/train_english.csv"  # 替换为您的训练数据文件路径
    formatted_data = prepare_data(train_file)
    
    # 2. 下载并设置模型
    model, tokenizer = setup_model(selected_model, use_4bit=True)
    
    # 3. 处理数据
    processed_datasets = process_data(formatted_data, tokenizer, selected_model)
    
    # 4. 训练模型
    model, tokenizer = train_model(model, tokenizer, processed_datasets)

if __name__ == "__main__":
    main()