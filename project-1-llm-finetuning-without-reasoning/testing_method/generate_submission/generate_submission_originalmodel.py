import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from huggingface_hub import login
from tqdm import tqdm

def load_model_from_huggingface(model_id, use_auth=False, token=None, use_4bit=False, use_8bit=False):
    """從Hugging Face載入模型
    
    Args:
        model_id: Hugging Face模型ID
        use_auth: 是否使用認證
        token: Hugging Face token
        use_4bit: 是否使用4位元量化
        use_8bit: 是否使用8位元量化
    
    Returns:
        model, tokenizer
    """
    # 設定Hugging Face認證（如果需要）
    if use_auth and token:
        login(token=token)
    
    print(f"從Hugging Face載入模型: {model_id}")
    
    # 載入tokenizer
    print("載入tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left"
    )
    
    # 確保tokenizer有padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 設定量化配置
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        print("使用4位元量化載入模型...")
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        print("使用8位元量化載入模型...")
    else:
        quantization_config = None
        print("使用完整精度載入模型...")
    
    # 載入模型
    try:
        # 判斷是否支援flash attention
        attn_implementation = None
        try:
            import flash_attn
            attn_implementation = "flash_attention_2"
            print("啟用Flash Attention 2")
        except ImportError:
            print("未安裝Flash Attention，使用標準注意力機制")
        
        model_load_args = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
        }
        
        # 如果有量化設定則添加
        if quantization_config:
            model_load_args["quantization_config"] = quantization_config
        
        # 如果支援flash attention則添加
        if attn_implementation:
            model_load_args["attn_implementation"] = attn_implementation
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_load_args
        )
        
    except Exception as e:
        print(f"模型載入失敗: {e}")
        print("嘗試使用較大超時時間重新載入...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            max_memory={0: "22GiB"},  # 限制GPU記憶體使用
            offload_folder="offload"  # 如果RAM不足可以使用磁盤卸載
        )
    
    print("模型載入完成!")
    model.eval()  # 設置為評估模式
    return model, tokenizer

def extract_option(text):
    """從文本中提取選項字母 (A, B, C, D)"""
    # 直接檢查開頭是否為答案選項
    if text and text[0] in "ABCD":
        return text[0]
    
    # 檢查"答案是X"的模式
    if "答案是A" in text or "答案為A" in text or "選A" in text:
        return "A"
    elif "答案是B" in text or "答案為B" in text or "選B" in text:
        return "B"
    elif "答案是C" in text or "答案為C" in text or "選C" in text:
        return "C"
    elif "答案是D" in text or "答案為D" in text or "選D" in text:
        return "D"
    
    # 檢查文本中的單獨字母
    for char in text:
        if char in "ABCD":
            return char
    
    return None

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
    
    # ===== 新增：模型預熱階段 =====
    print("開始模型預熱階段...")
    
    # 預熱提示設計 - 使用多種形式的預熱提示
    warmup_prompts = [
        # 第一種預熱：直接指導回答格式
        "<|im_start|>user\n請以單一字母(A、B、C或D)回答選擇題。<|im_end|>\n<|im_start|>assistant\n好的，我會以單一字母A、B、C或D回答問題。<|im_end|>",
        
        # 第二種預熱：提供範例問題和回答
        "<|im_start|>user\n以下是一道選擇題，請選擇正確答案：\n1+1=?\nA. 1\nB. 2\nC. 3\nD. 4\n答案是：<|im_end|>\n<|im_start|>assistant\nB<|im_end|>",
        
        # 第三種預熱：明確強調回答格式的重要性
        "<|im_start|>user\n回答選擇題時，請直接給出選項字母（A、B、C或D），不需要額外解釋。<|im_end|>\n<|im_start|>assistant\n明白，我會直接回答A、B、C或D，不會添加額外解釋。<|im_end|>"
    ]
    
    # 執行預熱
    with torch.no_grad():
        for i, prompt in enumerate(warmup_prompts):
            print(f"執行預熱提示 {i+1}/{len(warmup_prompts)}...")
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            # 使用短的生成參數進行預熱
            warmup_config = GenerationConfig(
                max_new_tokens=5,
                temperature=0.01,
                do_sample=False
            )
            _ = model.generate(**inputs, generation_config=warmup_config)
            print(f"預熱提示 {i+1} 完成")
    
    print("模型預熱完成！")
    # ===== 預熱階段結束 =====
    
    # 檢查是否只是提交模板（只有ID欄位）
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
        max_new_tokens=10,
        temperature=0.1,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1,
        do_sample=False  # 設為False以獲得更確定的答案
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
                prompt = f"<|im_start|>user\n請選擇以下問題的正確答案。請務必以單個字母(A、B、C或D)開始你的回答。\n\n{question}\n{options}\n答案是：<|im_end|>\n<|im_start|>assistant\n"
                
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        generation_config=generation_config
                    )
                
                # 解碼模型回答
                response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                
                # 從回答中提取選項字母
                answer = extract_option(response)
                
                if not answer:
                    print(f"警告：問題 {id_value} 無法提取明確答案，設為默認值X。模型回答: {response}")
                    answer = "X"
            else:
                # 如果只有ID，只能用默認答案
                print(f"ID {id_value}: 測試文件只包含ID，無問題內容。使用默認答案X")
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
    output_file = "submission.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"預測完成，結果已保存至 {output_file}")
    return results_df

def main():
    # 從Hugging Face載入模型
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # 可以替換為其他模型ID
    
    # 取消註釋並填入你的token以訪問限制性模型
    # hf_token = "your_huggingface_token"
    
    # 載入模型
    model, tokenizer = load_model_from_huggingface(
        model_id=model_id,
        use_auth=False,  # 設為True以使用認證
        token=None,      # 填入你的HF token
        use_4bit=False,  # 設為True以使用4位元量化節省記憶體
        use_8bit=False   # 設為True以使用8位元量化節省記憶體
    )
    
    # 生成預測
    test_file = "C:/Users/NTHUILST/Ray/DL/data/test-v2.csv"  # 請確保路徑正確
    generate_predictions(model, tokenizer, test_file)

if __name__ == "__main__":
    main()
