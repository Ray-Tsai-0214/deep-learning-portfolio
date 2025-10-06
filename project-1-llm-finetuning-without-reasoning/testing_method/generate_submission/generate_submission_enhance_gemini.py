import pandas as pd
import requests
import json
import time
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def call_gemini_api(prompt, api_key, model="google/gemini-2.0-flash-001"):
    """調用OpenRouter的Gemini API進行推理"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.1
    }
    
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", 
                                headers=headers, 
                                data=json.dumps(data))
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"Gemini API錯誤: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Gemini API調用錯誤: {e}")
        return None

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
            return "A"
    except Exception as e:
        print(f"API調用錯誤: {e}")
        return "A"

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
                results.append({"ID": ids[i], "answer": "A"})
    
    return results

def generate_predictions_batch(openai_api_key, gemini_api_key, test_file, batch_size=8):
    """使用Gemini生成回答，然後用OpenAI API提取答案"""
    checkpoint_file = "prediction_checkpoint.csv"
    final_output_file = "submission_gemini_enhanced.csv"
    
    # 加載測試數據
    test_df = pd.read_csv(test_file)
    print(f"測試檔案包含 {len(test_df)} 個問題")
    
    # 找到ID欄位
    id_col = "ID" if "ID" in test_df.columns else test_df.columns[0]
    
    # 檢查是否有檢查點文件
    processed_ids = set()
    results = []
    
    if os.path.exists(checkpoint_file):
        checkpoint_df = pd.read_csv(checkpoint_file)
        processed_ids = set(checkpoint_df[id_col].astype(str))
        results = checkpoint_df.to_dict('records')
        print(f"從檢查點恢復，已處理 {len(processed_ids)} 個問題")
    
    # 過濾掉已處理的問題
    remaining_indices = [
        i for i, row in test_df.iterrows() 
        if str(row[id_col]) not in processed_ids
    ]
    
    print(f"剩餘 {len(remaining_indices)} 個問題需要處理")
    print(f"使用批大小: {batch_size}，預計批次數: {len(remaining_indices)//batch_size + 1}")
    
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

Please select one final answer (A, B, C, or D) and provide a brief explanation. Remember that when uncertain, especially between option A and any other option, choose A.
"""
                batch_prompts.append(prompt)
            
            # 使用ThreadPoolExecutor並行調用Gemini API
            batch_responses = []
            with ThreadPoolExecutor(max_workers=min(batch_size, 5)) as executor:
                futures = [executor.submit(call_gemini_api, prompt, gemini_api_key) for prompt in batch_prompts]
                for future in futures:
                    response = future.result()
                    if response:
                        batch_responses.append(response)
                    else:
                        batch_responses.append("Unable to generate response, default to answer A.")
            
            # 確保batch_responses的長度與batch_ids匹配
            while len(batch_responses) < len(batch_ids):
                batch_responses.append("No response generated. Default to answer A.")
                
            # 顯示樣本
            if batch_start == 0:
                print(f"\n示例問題回應: {batch_responses[0][:200]}...\n")
            
            # 使用OpenAI API批量處理回應
            print(f"批次 {batch_start//batch_size + 1}: 調用API解析 {len(batch_responses)} 個回應...")
            batch_results = batch_process_with_api(batch_responses, batch_ids, openai_api_key, max_workers=min(10, len(batch_responses)))
            results.extend(batch_results)
            
            # 定期保存檢查點
            current_time = time.time()
            if (current_time - last_checkpoint_time > 60) or (batch_start + batch_size >= len(remaining_indices)):
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_csv(checkpoint_file, index=False)
                last_checkpoint_time = current_time
                
                elapsed_time = current_time - start_time
                total_processed = len(results)
                questions_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
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
    # API密鑰
    openai_api_key = "YOUR_OPENAI_API_KEY_HERE"
    gemini_api_key = "YOUR_GEMINI_API_KEY_HERE" # 請替換為您的OpenRouter API密鑰
    
    # 批處理大小 - 適合API調用的批大小
    batch_size = 5  # API調用批大小通常需要小一些
    
    print(f"使用Google Gemini 2.0 Flash通過OpenRouter進行推理")
    print(f"選擇批大小: {batch_size}")
    
    # 生成預測
    test_file = "./data/test-v2_english.csv"
    generate_predictions_batch(openai_api_key, gemini_api_key, test_file, batch_size)

if __name__ == "__main__":
    main()