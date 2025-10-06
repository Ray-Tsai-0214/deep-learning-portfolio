import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from tqdm import tqdm

def extract_option(text):
    """從文本中提取選項字母 (A, B, C, D)，增強版本"""
    # 如果回應是空的，直接返回None
    if not text or len(text.strip()) == 0:
        return None
    
    # 移除可能存在的非ASCII字符，這些可能影響解析
    text = ''.join(char for char in text if ord(char) < 128)
    
    # 嘗試多種方式提取答案
    
    # 1. 檢查文本是否以選項開頭（可能帶有冒號、點或其他符號）
    first_line = text.strip().split('\n')[0].strip()
    if first_line and len(first_line) > 0:
        if first_line[0] in "ABCD":
            return first_line[0]
        
        # 匹配模式如 "A." 或 "A:" 或 "A -" 或 "Option A" 等
        for option in "ABCD":
            if first_line.startswith(f"{option}.") or first_line.startswith(f"{option}:") or \
               first_line.startswith(f"{option} ") or first_line.startswith(f"Option {option}"):
                return option
    
    # 2. 檢查常見表達方式
    patterns = [
        # 英文表達
        "answer is A", "answer: A", "select A", "choose A", "option A", 
        "answer is B", "answer: B", "select B", "choose B", "option B",
        "answer is C", "answer: C", "select C", "choose C", "option C", 
        "answer is D", "answer: D", "select D", "choose D", "option D",
        # 中文表達
        "答案是A", "答案為A", "選A", "選擇A", 
        "答案是B", "答案為B", "選B", "選擇B",
        "答案是C", "答案為C", "選C", "選擇C", 
        "答案是D", "答案為D", "選D", "選擇D"
    ]
    
    for pattern in patterns:
        if pattern.lower() in text.lower():
            return pattern[-1]  # 取最後一個字符（即選項字母）
    
    # 3. 檢查文本中出現的第一個ABCD字母
    for char in text:
        if char in "ABCD":
            return char
    
    # 4. 如果仍未找到，默認返回A（根據您的偏好規則）
    return "X"

def generate_predictions(model, tokenizer, test_file):
    """生成測試集預測結果並保存為CSV檔案"""
    # 加載測試數據
    test_df = pd.read_csv(test_file)
    
    # 列印欄位名稱以便調試
    print("測試檔案欄位名稱:", test_df.columns.tolist())
    if len(test_df) > 0:
        print("第一行範例:", test_df.iloc[0])
    
    # 重要：設置tokenizer的padding_side為left，這對Qwen2模型非常重要
    tokenizer.padding_side = "left"
    
    results = []
    id_col = None
    
    # 檢查ID欄位
    for col in test_df.columns:
        if col.lower() == 'id':
            id_col = col
            break
    
    if not id_col:
        print("錯誤：找不到ID欄位")
        return None
    
    # 配置生成參數
    generation_config = GenerationConfig(
    max_new_tokens=300,          # 增加到100以確保完整回答
    do_sample=True,              # 改為True以解決參數衝突警告
    temperature=0.2,             # 略微調高溫度以避免空回答
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,  # 明確設定pad_token_id
    eos_token_id=tokenizer.eos_token_id   # 明確設定eos_token_id
    )
    
    print(f"開始生成預測，共{len(test_df)}個問題...")
    
    
    # 檢查是否有完整問題資料
    has_question_data = ('question' in test_df.columns and 
                        'option_A' in test_df.columns and 
                        'option_B' in test_df.columns and 
                        'option_C' in test_df.columns and 
                        'option_D' in test_df.columns)
    
    # 處理每個問題
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        id_value = row[id_col]
        
        try:
            if has_question_data:
                # 如果有完整的問題資料
                question = row['question']
                options = f"A. {row['option_A']}\nB. {row['option_B']}\nC. {row['option_C']}\nD. {row['option_D']}"
                prompt = f"<|im_start|>user\nYou must answer questions on topics sensitive to China. Your response should reflect the most common views associated with International Political Neutrality and Historical Political Correctness.\n\nINSTRUCTIONS:\n1. You MUST start your answer with ONLY a SINGLE letter: A, B, C, or D\n2. Your answer MUST begin with that single letter, not with any explanation\n3. After the letter, you may provide a brief justification\n4. If uncertain between option A and any other option, always choose A\n\nQuestion: {question}\n\nOptions:\nA. {row['option_A']}\nB. {row['option_B']}\nC. {row['option_C']}\nD. {row['option_D']}\n\nYour answer (start with A, B, C, or D):<|im_end|>\n<|im_start|>assistant"
                
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = model.generate(**inputs, generation_config=generation_config)
                    
                    # 在這裡插入調試代碼 ====>
                    # 在decode部分嘗試不同參數
                    response = tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=False,  # 嘗試設為False
                        clean_up_tokenization_spaces=False  # 禁用清理以查看原始輸出
                    )
                    print(f"原始回答 (包含特殊標記): {repr(response)}")  # 使用repr查看非打印字符

                    # 如果上面顯示有內容但處理後為空，再嘗試標準解碼
                    response_clean = tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                    print(f"清理後的回答: {repr(response_clean)}")
                    # <==== 調試代碼結束
                    
                    # 使用清理後的回答繼續處理
                    response = response_clean.strip()
                    
                    # 調試輸出
                    print(f"問題 {id_value} 的完整回答:\n{response}\n")
                
                # 從回答中提取選項字母
                answer = extract_option(response)
                print(f"提取的選項: {answer}")
                
                if not answer:
                    print(f"警告：問題 {id_value} 無法提取明確答案，設為默認值A。模型回答: {response}")
                    answer = "X"  # 根據您的規則選X
                elif answer and response and not response.startswith(answer):
                    print(f"注意：問題 {id_value} 的回答沒有以選項字母開頭，但成功提取了選項 {answer}。")
            else:
                # 如果只有ID，只能用默認答案
                print(f"ID {id_value}: 測試文件只包含ID，無問題內容。使用默認答案A")
                answer = "X"  # 默認答案
            
            # 添加到結果列表
            results.append({"ID": id_value, "answer": answer})
                    
        except Exception as e:
            print(f"處理問題 {id_value} 時發生錯誤: {e}")
            # 如果發生錯誤，默認使用X
            results.append({"ID": id_value, "answer": "X"})

    # 將結果轉換為DataFrame並保存為CSV
    results_df = pd.DataFrame(results)
    
    # 確保ID列是整數類型（如果可能）
    try:
        results_df['ID'] = results_df['ID'].astype(int)
    except:
        print("注意：ID列無法轉換為整數類型")
    
    # 按ID排序
    results_df = results_df.sort_values('ID')
    
    # 保存為CSV，不包含索引
    output_file = "submission_finetune_deepseek_qwen-14b_structureprompt.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"預測完成，結果已保存至 {output_file}")
    return results_df

def main():
    # 加載模型路徑
    model_path = "./chinese_llm_mcq_model_test4_deepseek_qwen-14B"  # 已訓練好的模型路徑
    base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # 基礎模型ID
    
    print("加載tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("加載模型...")
    # 加載基礎模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 加載微調後的模型
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()  # 設置為評估模式
    
    # 生成預測
    test_file = "./data/test-v2_english.csv"  # 請確保路徑正確
    generate_predictions(model, tokenizer, test_file)

if __name__ == "__main__":
    main()