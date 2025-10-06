#!/usr/bin/env python3
"""
GRPO模型測試提交生成腳本
使用訓練好的GRPO模型生成競賽提交檔案
"""

import os
import torch
import pandas as pd
import re
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel

def setup_model_and_tokenizer(model_path):
    """設置模型和tokenizer"""
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    # 量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # 載入基礎模型
    print("📦 載入基礎模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    # 載入GRPO微調的模型
    print("🔧 載入GRPO微調模型...")
    if os.path.exists(model_path):
        model = PeftModel.from_pretrained(base_model, model_path)
        print(f"✅ 成功載入GRPO模型: {model_path}")
    else:
        print(f"⚠️ GRPO模型路徑不存在: {model_path}")
        print("使用基礎模型進行推理...")
        model = base_model
    
    # 載入tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def load_test_data():
    """載入測試數據"""
    test_file = "../data/test-check-v2.csv"
    
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        print(f"✅ 成功載入測試數據: {len(df)} 題")
        return df
    else:
        print(f"❌ 測試檔案不存在: {test_file}")
        # 創建示例測試數據
        sample_data = {
            'id': [i for i in range(100)],
            'question': [f'測試問題 {i}' for i in range(100)],
            'option_A': [f'選項A{i}' for i in range(100)],
            'option_B': [f'選項B{i}' for i in range(100)],
            'option_C': [f'選項C{i}' for i in range(100)],
            'option_D': [f'選項D{i}' for i in range(100)]
        }
        return pd.DataFrame(sample_data)

def format_prompt(row):
    """格式化輸入提示"""
    return f"""問題：{row['question']}

選項：
A. {row['option_A']}
B. {row['option_B']}  
C. {row['option_C']}
D. {row['option_D']}

請選擇正確答案並說明理由。"""

def extract_answer(text):
    """從回答中提取答案"""
    # 清理文本
    text = text.strip()
    
    # 多種答案提取模式
    patterns = [
        r'答案[：:]\s*([ABCD])',
        r'選擇\s*([ABCD])',
        r'答案是\s*([ABCD])',
        r'正確答案是\s*([ABCD])',
        r'^([ABCD])',
        r'([ABCD])\s*[。.]',
        r'選項\s*([ABCD])',
        r'([ABCD])\s*是正確的'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).upper()
            if answer in ['A', 'B', 'C', 'D']:
                return answer
    
    # 如果都找不到，嘗試從開頭查找
    for char in text:
        if char.upper() in ['A', 'B', 'C', 'D']:
            return char.upper()
    
    # 預設返回A
    return 'A'

def generate_answer(model, tokenizer, prompt, max_retries=3):
    """生成答案"""
    generation_config = GenerationConfig(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )
    
    for attempt in range(max_retries):
        try:
            # 編碼輸入
            inputs = tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024,
                padding=True
            ).to(model.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    generation_config=generation_config
                )
            
            # 解碼輸出
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取新生成的部分
            response = full_response[len(prompt):].strip()
            
            # 提取答案
            answer = extract_answer(response)
            
            if answer in ['A', 'B', 'C', 'D']:
                return answer, response
            
        except Exception as e:
            print(f"生成失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                return 'A', "生成失敗"
    
    return 'A', "生成失敗"

def main():
    """主要執行流程"""
    print("🎯 GRPO模型測試提交生成")
    print("=" * 50)
    
    # 設置模型路徑
    model_path = "../models/grpo_chinese_50percent_0624/final_model"
    
    # 載入模型
    model, tokenizer = setup_model_and_tokenizer(model_path)
    
    # 載入測試數據
    test_df = load_test_data()
    
    # 準備結果
    results = []
    total_questions = len(test_df)
    
    print(f"🔍 開始處理 {total_questions} 個問題...")
    
    # 處理每個問題
    for idx, row in test_df.iterrows():
        if idx % 10 == 0:
            print(f"📊 進度: {idx}/{total_questions} ({idx/total_questions*100:.1f}%)")
        
        # 格式化提示
        prompt = format_prompt(row)
        
        # 生成答案
        answer, response = generate_answer(model, tokenizer, prompt)
        
        # 保存結果
        results.append({
            'id': row['id'],
            'answer': answer,
            'response': response[:200] + "..." if len(response) > 200 else response
        })
    
    # 創建提交檔案
    submission_df = pd.DataFrame([
        {'id': result['id'], 'answer': result['answer']} 
        for result in results
    ])
    
    # 保存提交檔案
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"../submission/submission_grpo_{timestamp}.csv"
    
    os.makedirs("../submission", exist_ok=True)
    submission_df.to_csv(submission_file, index=False)
    
    print(f"✅ 提交檔案已生成: {submission_file}")
    
    # 答案分布統計
    answer_counts = submission_df['answer'].value_counts()
    print("\n📊 答案分布:")
    for answer, count in answer_counts.items():
        percentage = count / len(submission_df) * 100
        print(f"  {answer}: {count} ({percentage:.1f}%)")
    
    # 保存詳細結果（包含回答內容）
    detailed_file = f"../submission/detailed_results_grpo_{timestamp}.csv"
    detailed_df = pd.DataFrame(results)
    detailed_df.to_csv(detailed_file, index=False, encoding='utf-8-sig')
    
    print(f"📋 詳細結果已保存: {detailed_file}")
    print("🎉 測試提交生成完成！")

if __name__ == "__main__":
    main()