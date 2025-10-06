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
    "qwen1.5_14b": "Qwen/Qwen1.5-14B",
    "yi_6b": "01-ai/Yi-6B",
    "deepseek_7b": "deepseek-ai/deepseek-llm-7b-base",
    "deepseek_v2": "deepseek-ai/deepseek-v2",
    "baichuan2_7b": "baichuan-inc/Baichuan2-7B-Base",
    "internlm2_7b": "internlm/internlm2-7b"
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
        
        formatted_example = {
            "instruction": f"請選擇以下問題的正確答案。\n\n{question}\n{options}",
            "input": "",
            "output": row[answer_col]
        }
        formatted_data.append(formatted_example)
    
    # 保存為JSONL格式用於訓練
    with open('formatted_train_data.jsonl', 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return formatted_data

# 2. 从Hugging Face下载并设置模型
def setup_model(model_name_or_key="qwen_QwQ", use_4bit=True):
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
        padding_side="right"
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
    
    # 定义数据预处理函数
    def preprocess_function(examples):
    # 构建提示模板
        prompts = []
        targets = []

        # 根據模型選擇對話模板
        # 為所有輸入構建提示，不依賴於模型類型
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            full_prompt = f"<|im_start|>user\n{instruction}"
            if input_text:
                full_prompt += f"\n{input_text}"
            full_prompt += "<|im_end|>\n<|im_start|>assistant\n"
            
            prompts.append(full_prompt)
            targets.append(f"{output}<|im_end|>")
        
        # 確保prompts和targets不為空
        if not prompts or not targets:
            # 如果列表為空，添加一個占位符以避免出錯
            print("警告：沒有有效的提示或目標")
            prompts = ["<|im_start|>user\n佔位符問題<|im_end|>\n<|im_start|>assistant\n"]
            targets = ["佔位符回答<|im_end|>"]
            
        # 编码输入
        tokenized_inputs = tokenizer(
            prompts, 
            truncation=True,
            max_length=256,
            padding=False,
            return_tensors=None
        )
        
        # 编码目标输出
        tokenized_targets = tokenizer(
            targets,
            truncation=True,
            max_length=96,
            padding=False,
            return_tensors=None
        )
        
        # 构建输入输出对
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for input_ids, target_ids in zip(tokenized_inputs["input_ids"], tokenized_targets["input_ids"]):
            # 组合输入和输出
            combined_input_ids = input_ids + target_ids
            attention_mask = [1] * len(combined_input_ids)
            
            # 标签: 输入部分为-100，输出部分为实际token ID
            labels = [-100] * len(input_ids) + target_ids
            
            all_input_ids.append(combined_input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)
        
        # 用于确保每个batch中的序列长度相同
        result = {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels
        }
        
        return result
    
    # 应用预处理
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=8,  # 使用多进程加速
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
    model.save_pretrained("./chinese_llm_mcq_model")
    tokenizer.save_pretrained("./chinese_llm_mcq_model")
    print("模型训练完成并保存！")
    
    return model, tokenizer

# # 5. 推理与预测函数
# def generate_predictions(model, tokenizer, test_file):
#     """生成測試集預測結果並保存為CSV檔案"""
#     # 加載測試數據
#     test_df = pd.read_csv(test_file)
#     results = []
    
#     # 配置生成參數
#     generation_config = GenerationConfig(
#         max_new_tokens=10,  # 只需要短回答
#         temperature=0.1,    # 低溫度以獲得確定性輸出
#         top_p=0.95,
#         top_k=40,
#         repetition_penalty=1.1,
#         do_sample=False     # 關閉採樣以獲得更確定的答案
#     )
    
#     print(f"開始生成預測，共{len(test_df)}個問題...")
    
#     for idx, row in test_df.iterrows():
#         question = row['question']
#         options = f"A. {row['option_A']}\nB. {row['option_B']}\nC. {row['option_C']}\nD. {row['option_D']}"
        
#         prompt = f"<|im_start|>user\n請選擇以下問題的正確答案。\n\n{question}\n{options}\n答案是：<|im_end|>\n<|im_start|>assistant\n"
        
#         inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
#         try:
#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     generation_config=generation_config
#                 )
            
#             # 解碼模型回答
#             response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            
#             # 從回答中提取選項字母
#             answer = extract_option(response)
            
#             # 默認回答為A，以防無法提取
#             if not answer:
#                 print(f"警告：問題 {row['ID']} 無法提取明確答案，設為默認值A。模型回答: {response}")
#                 answer = "A"
                
#             # 添加到結果列表
#             results.append({"ID": row['ID'], "answer": answer})
            
#             # 顯示進度
#             if (idx + 1) % 10 == 0 or idx == len(test_df) - 1:
#                 print(f"已完成: {idx + 1}/{len(test_df)} 個預測")
                
#         except Exception as e:
#             print(f"處理問題 {row['ID']} 時發生錯誤: {e}")
#             # 如果發生錯誤，默認使用A
#             results.append({"ID": row['ID'], "answer": "A"})
    
#     # 將結果轉換為DataFrame並保存為CSV
#     results_df = pd.DataFrame(results)
    
#     # 確保ID列是整數類型
#     results_df['ID'] = results_df['ID'].astype(int)
    
#     # 按ID排序
#     results_df = results_df.sort_values('ID')
    
#     # 保存為CSV，不包含索引
#     output_file = "submission.csv"
#     results_df.to_csv(output_file, index=False)
    
#     print(f"預測完成，結果已保存至 {output_file}")
#     return results_df

# def extract_option(text):
#     """從文本中提取選項字母 (A, B, C, D)"""
#     # 直接檢查開頭是否為答案選項
#     if text and text[0] in "ABCD":
#         return text[0]
    
#     # 檢查"答案是X"的模式
#     if "答案是A" in text or "答案為A" in text or "選A" in text:
#         return "A"
#     elif "答案是B" in text or "答案為B" in text or "選B" in text:
#         return "B"
#     elif "答案是C" in text or "答案為C" in text or "選C" in text:
#         return "C"
#     elif "答案是D" in text or "答案為D" in text or "選D" in text:
#         return "D"
    
#     # 檢查文本中的單獨字母
#     for char in text:
#         if char in "ABCD":
#             return char
    
#     return None
    
# 主函数
def main():
    # 选择要使用的模型
    selected_model = "qwen_QwQ"  # 可以从CHINESE_MODELS中选择
    
    # 1. 准备训练数据
    train_file = "C:/Users/NTHUILST/Ray/DL/data/train.csv"  # 替换为您的训练数据文件路径
    formatted_data = prepare_data(train_file)
    
    # 2. 下载并设置模型
    model, tokenizer = setup_model(selected_model, use_4bit=True)
    
    # 3. 处理数据
    processed_datasets = process_data(formatted_data, tokenizer, selected_model)
    
    # 4. 训练模型
    model, tokenizer = train_model(model, tokenizer, processed_datasets)

    # # 5. 生成測試集
    # test_file = "C:/Users/NTHUILST/Ray/DL/data/test-v2.csv"
    # generate_predictions(model, tokenizer, test_file)

if __name__ == "__main__":
    main()
