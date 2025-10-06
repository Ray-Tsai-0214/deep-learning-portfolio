import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from tqdm import tqdm
import time
import os
import gc
import requests
import json
from concurrent.futures import ThreadPoolExecutor

def call_mini_api(prompt, api_key, model="gpt-4o-mini"):
    """調用OpenAI API獲取選項判斷結果"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一個專門從複雜文本中提取多選題答案的助手。你只需回答A、B、C或D，不需要任何解釋。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 5,
        "temperature": 0.1
    }
    
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", 
                              headers=headers, 
                              data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            answer_text = result["choices"][0]["message"]["content"].strip().upper()
            # 提取第一個ABCD字符
            for char in answer_text:
                if char in "ABCD":
                    return char
            return "A"  # 默認值
        else:
            print(f"API錯誤: {response.status_code} - {response.text}")
            return "X"
    except Exception as e:
        print(f"API調用錯誤: {e}")
        return "X"

def batch_process_with_api(responses, ids, api_key, max_workers=10):
    """批量使用API處理多個回應"""
    prompts = []
    for response in responses:
        prompt = f"""The following is a response to a multiple-choice question. Please extract option A, B, C, or D from it. If uncertain, default to A.
                    Response text:
                    {response}
                    Please respond with only one letter (A, B, C, or D):"""
        prompts.append(prompt)
    
    results = []
    
    # 使用線程池並行處理API請求
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        api_futures = [executor.submit(call_mini_api, prompt, api_key) for prompt in prompts]
        for i, future in enumerate(api_futures):
            try:
                answer = future.result()
                results.append({"ID": ids[i], "answer": answer})
            except Exception as e:
                print(f"處理ID {ids[i]}時出錯: {e}")
                results.append({"ID": ids[i], "answer": "X"})
    
    return results

def generate_predictions_batch(model, tokenizer, test_file, api_key, batch_size=8):
    """使用DeepSeek生成回答，然後用API提取答案"""
    checkpoint_file = "prediction_checkpoint.csv"
    final_output_file = "submission_api_enhanced.csv"
    
    # 加載測試數據
    test_df = pd.read_csv(test_file)
    print(f"測試檔案包含 {len(test_df)} 個問題")
    
    # 設置tokenizer
    tokenizer.padding_side = "left"
    id_col = "ID" if "ID" in test_df.columns else test_df.columns[0]
    
    # 檢查是否有檢查點文件
    processed_ids = set()
    results = []
    
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        processed_ids = set(checkpoint_df[id_col].astype(str))
        results = checkpoint_df.to_dict('records')
        print(f"從檢查點恢復，已處理 {len(processed_ids)} 個問題")
    
    # 優化的生成參數 - 生成較詳細的回答讓API判斷
    generation_config = GenerationConfig(
        max_new_tokens=300,           # 生成足夠的內容供API判斷
        do_sample=False,              # 確定性輸出
        num_beams=1,                  # 貪婪解碼
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    # 過濾掉已處理的問題
    remaining_indices = [
        i for i, row in test_df.iterrows() 
        if str(row[id_col]) not in processed_ids
    ]
    
    print(f"剩餘 {len(remaining_indices)} 個問題需要處理")
    print(f"使用批大小: {batch_size}，預計批次數: {len(remaining_indices)//batch_size + 1}")
    
    # 獲取GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"使用GPU: {gpu_name}")
    
    # 計時
    start_time = time.time()
    last_checkpoint_time = start_time
    
    # 批處理問題
    for batch_start in tqdm(range(0, len(remaining_indices), batch_size)):
        batch_end = min(batch_start + batch_size, len(remaining_indices))
        batch_indices = remaining_indices[batch_start:batch_end]
        batch_rows = [test_df.iloc[i] for i in batch_indices]
        batch_ids = [row[id_col] for row in batch_rows]
        
        try:
            # 準備批量輸入
            batch_prompts = []
            for row in batch_rows:
                prompt = f"""Please analyze this multiple-choice question in detail and select the most appropriate answer:

                Question: {row['question']}

                A: {row['option_A']}
                B: {row['option_B']}
                C: {row['option_C']}
                D: {row['option_D']}

                Please select one final answer (A, B, C, or D):"""
                batch_prompts.append(prompt)
            
            # 批量編碼
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")
            
            # 批量生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # 解碼響應
            batch_responses = []
            for i in range(len(batch_indices)):
                output_ids = outputs[i][inputs.input_ids.shape[1]:]
                response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                batch_responses.append(response)
                
                # 顯示樣本
                if batch_start == 0 and i == 0:
                    print(f"\n示例問題回應: {response[:100]}...\n")
            
            # 使用API批量處理回應
            print(f"批次 {batch_start//batch_size + 1}: 調用API解析 {len(batch_responses)} 個回應...")
            batch_results = batch_process_with_api(batch_responses, batch_ids, api_key, max_workers=min(10, len(batch_responses)))
            results.extend(batch_results)
            
            # 定期保存檢查點
            current_time = time.time()
            if (current_time - last_checkpoint_time > 120) or (batch_start + batch_size >= len(remaining_indices)):
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_csv(checkpoint_file, index=False)
                last_checkpoint_time = current_time
                
                elapsed_time = current_time - start_time
                total_processed = len(results)
                questions_per_second = total_processed / elapsed_time
                remaining_questions = len(test_df) - total_processed
                estimated_time_left = remaining_questions / questions_per_second if questions_per_second > 0 else 0
                
                print(f"\n進度: {total_processed}/{len(test_df)} 題 ({total_processed/len(test_df)*100:.1f}%)")
                print(f"速度: {questions_per_second:.2f} 題/秒")
                print(f"已用時間: {elapsed_time/60:.1f}分鐘, 預計剩餘: {estimated_time_left/60:.1f}分鐘")
        
        except Exception as e:
            print(f"批處理錯誤 (IDs {batch_ids[0]}-{batch_ids[-1]}): {e}")
            # 出錯時使用默認答案A
            for id_value in batch_ids:
                results.append({id_col: id_value, "answer": "A"})
            
            # 保存檢查點
            checkpoint_df = pd.DataFrame(results)
            checkpoint_df.to_csv(checkpoint_file, index=False)
            last_checkpoint_time = time.time()
    
    # 總計時間和統計
    total_time = time.time() - start_time
    processed_count = len(results)
    questions_per_second = processed_count / total_time if total_time > 0 else 0
    print(f"\n總處理時間: {total_time/60:.2f}分鐘")
    print(f"平均速度: {questions_per_second:.2f} 題/秒")
    
    # 統計答案分布
    answer_counts = {}
    for result in results:
        answer = result["answer"]
        if answer in answer_counts:
            answer_counts[answer] += 1
        else:
            answer_counts[answer] = 1
    
    print("\n答案分布:")
    for answer, count in sorted(answer_counts.items()):
        print(f"{answer}: {count} 題 ({count/len(results)*100:.1f}%)")
    
    # 生成最終CSV
    final_df = pd.DataFrame(results)
    
    # 確保ID列格式正確
    try:
        final_df[id_col] = final_df[id_col].astype(int)
    except:
        pass
    
    # 按ID排序
    final_df = final_df.sort_values(id_col)
    
    # 保存最終結果
    final_df.to_csv(final_output_file, index=False)
    
    print(f"預測完成，結果已保存至 {final_output_file}")
    return final_df

def main():
    # 您的OpenAI API密鑰
    api_key = "YOUR_OPENAI_API_KEY_HERE"  # 請替換為您的API密鑰
    
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
    print("加載tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    print("加載模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # 設置為評估模式並釋放內存
    model.eval()
    torch.cuda.empty_cache()

    # 根據GPU記憶體決定批大小
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
        free_memory_gb = free_memory / (1024**3)
        
        if free_memory_gb > 20:
            batch_size = 10
        elif free_memory_gb > 15:
            batch_size = 8
        elif free_memory_gb > 10:
            batch_size = 6
        else:
            batch_size = 4
    else:
        batch_size = 1
    
    print(f"選擇批大小: {batch_size}")
    
    # 生成預測
    test_file = "./data/test-v2_english.csv"
    generate_predictions_batch(model, tokenizer, test_file, api_key, batch_size)

if __name__ == "__main__":
    main()