import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import csv
from tqdm import tqdm
import time
import re
import os

class ReasoningModelPredictor:
    def __init__(self, model_path="./chinese_llm_mcq_model_qwen2.5_7b_reasoning"):
        """初始化推理增強微調模型進行預測"""
        print("🔄 加載推理增強微調模型...")
        
        try:
            # 加載基礎模型
            print("   - 加載基礎模型 Qwen2.5-7B-Instruct...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加載微調適配器
            print("   - 加載推理增強微調適配器...")
            self.model = PeftModel.from_pretrained(self.base_model, model_path)
            
            # 加載tokenizer
            print("   - 加載tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            print("✅ 推理增強模型加載完成！")
            
        except Exception as e:
            print(f"❌ 模型加載失敗: {e}")
            raise e
    
    def predict_single_reasoning(self, question, option_a, option_b, option_c, option_d, mode="reasoning"):
        """對單個問題進行推理增強預測"""
        
        # 構建選項文本
        options = f"A. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}"
        
        # 根據模式選擇不同的prompt格式
        if mode == "reasoning":
            # 完整推理模式
            system_message = "你是一个善于分析和推理的助手。在回答选择题时，请提供清晰的思考过程和推理步骤，最后必须明确给出最终答案。"
            user_message = f"请分析以下选择题，提供详细的推理过程，然后给出最终答案。请在最后明确写出'最终答案：X'（X为A、B、C或D）。\n\n问题：{question}\n\n选项：\n{options}"
        elif mode == "simple":
            # 簡單模式（兼容基礎訓練）
            system_message = "你是一个提供有关中国敏感话题客观答案的助手。始终呈现多元视角并保持政治中立。"
            user_message = f"请回答以下选择题。你的回答必须仅以单个字母（A、B、C或D）开头。\n\n问题：{question}\n\n选项：\n{options}"
        else:
            # 混合模式
            system_message = "你是一个提供客观答案的助手。请简要说明你的推理过程，然后给出答案。"
            user_message = f"请回答以下选择题并简要说明理由。最后请明确写出'答案：X'（X为A、B、C或D）。\n\n问题：{question}\n\n选项：\n{options}"
        
        # 構建messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            # Tokenize
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # 生成回答（根據模式調整參數）
            max_tokens = 512 if mode == "reasoning" else 48  # 增加推理模式的token數量
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            # 提取答案字母（多層次提取策略）
            predicted_answer = self.extract_answer_from_response(response)
            
            return predicted_answer, response.strip()
            
        except Exception as e:
            print(f"❌ 預測失敗: {e}")
            return 'A', f"錯誤: {str(e)}"  # 默認返回A
    
    def extract_answer_from_response(self, response):
        """多層次答案提取策略，專為推理增強模型設計"""
        
        # 策略1: 查找 "最終答案" 或 "答案" 後的字母
        final_answer_patterns = [
            r'最終答案[：:]\s*([ABCD])',
            r'答案[：:]\s*([ABCD])', 
            r'正確答案[：:]\s*([ABCD])',
            r'選擇[：:]\s*([ABCD])',
            r'因此答案是[：:]?\s*([ABCD])',
            r'所以答案是[：:]?\s*([ABCD])',
            r'我的答案是[：:]?\s*([ABCD])',
            r'綜合分析.*答案是[：:]?\s*([ABCD])',
        ]
        
        for pattern in final_answer_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # 策略2: 查找推理結論中的答案
        reasoning_conclusion_patterns = [
            r'因此選擇\s*([ABCD])',
            r'所以選擇\s*([ABCD])',
            r'綜合以上.*選擇\s*([ABCD])',
            r'基於以上分析.*([ABCD])',
            r'結論是\s*([ABCD])',
        ]
        
        for pattern in reasoning_conclusion_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # 策略3: 查找回應開頭的字母（簡單模式）
        first_char_match = re.match(r'^\s*([ABCD])', response)
        if first_char_match:
            return first_char_match.group(1)
        
        # 策略4: 查找任何單獨出現的選項字母（最後出現的）
        standalone_letters = re.findall(r'\b([ABCD])\b', response)
        if standalone_letters:
            # 選擇最後一個（通常是最終答案）
            return standalone_letters[-1]
        
        # 策略5: 在推理過程中查找選項引用
        option_references = re.findall(r'選項\s*([ABCD])', response)
        if option_references:
            return option_references[-1]
        
        # 策略6: 查找任何包含ABCD的模式（作為最後手段）
        any_letter_match = re.search(r'([ABCD])', response)
        if any_letter_match:
            return any_letter_match.group(1)
        
        # 策略7: 基於推理內容的智能判斷
        # 如果推理過程明確否定某些選項，選擇剩下的
        rejected_options = set()
        rejection_patterns = [
            r'選項\s*([ABCD])\s*不正確',
            r'選項\s*([ABCD])\s*錯誤',
            r'([ABCD])\s*選項.*不符合',
            r'排除\s*([ABCD])',
        ]
        
        for pattern in rejection_patterns:
            matches = re.findall(pattern, response)
            rejected_options.update(matches)
        
        # 如果只剩下一個選項，返回它
        all_options = {'A', 'B', 'C', 'D'}
        remaining_options = all_options - rejected_options
        if len(remaining_options) == 1:
            return list(remaining_options)[0]
        
        # 如果都沒找到，返回A作為默認值
        print(f"⚠️  無法解析答案: {response[:200]}..., 默認返回A")
        return 'A'

def predict_test_data_with_reasoning(test_csv_path, model_path, output_path, prediction_mode="reasoning"):
    """對測試數據進行推理增強批量預測"""
    
    print("=" * 70)
    print("🧠 開始對測試數據進行推理增強預測")
    print(f"🎯 預測模式: {prediction_mode}")
    print("=" * 70)
    
    # 1. 加載測試數據
    print("📊 加載測試數據...")
    try:
        df = pd.read_csv(test_csv_path)
        print(f"✅ 成功加載 {len(df)} 個測試樣本")
        
        # 檢查數據格式並支援多種欄位名稱
        possible_id_cols = ['ID', 'id', 'Id']
        possible_question_cols = ['Question', 'question', '題目', '问题']
        possible_option_cols = {
            'A': ['Option A', 'option_A', '選項A', '选项A'],
            'B': ['Option B', 'option_B', '選項B', '选项B'],
            'C': ['Option C', 'option_C', '選項C', '选项C'], 
            'D': ['Option D', 'option_D', '選項D', '选项D']
        }
        
        # 自動檢測欄位名稱
        id_col = None
        for col in possible_id_cols:
            if col in df.columns:
                id_col = col
                break
        
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
        
        print(f"   檢測到的欄位: ID={id_col}, 問題={question_col}")
        print(f"   選項欄位: {option_cols}")
        
        if not all([id_col, question_col]) or len(option_cols) != 4:
            print(f"❌ 缺少必要欄位，可用欄位: {list(df.columns)}")
            return
            
    except Exception as e:
        print(f"❌ 加載測試數據失敗: {e}")
        return
    
    # 2. 初始化推理增強模型
    try:
        predictor = ReasoningModelPredictor(model_path)
    except Exception as e:
        print(f"❌ 模型初始化失敗: {e}")
        return
    
    # 3. 開始批量預測
    print(f"🔬 開始推理預測 {len(df)} 個問題...")
    results = []
    detailed_results = []  # 保存詳細推理過程
    
    # 添加進度條
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="推理預測進度"):
        try:
            question_id = int(row[id_col])
            question = str(row[question_col]).strip()
            option_a = str(row[option_cols['A']]).strip()
            option_b = str(row[option_cols['B']]).strip()
            option_c = str(row[option_cols['C']]).strip()
            option_d = str(row[option_cols['D']]).strip()
            
            # 進行推理預測
            predicted_answer, reasoning_process = predictor.predict_single_reasoning(
                question, option_a, option_b, option_c, option_d, mode=prediction_mode
            )
            
            results.append({
                'ID': question_id,
                'Answer': predicted_answer
            })
            
            # 保存詳細推理過程（用於分析）
            detailed_results.append({
                'ID': question_id,
                'Question': question,
                'Answer': predicted_answer,
                'Reasoning': reasoning_process
            })
            
            # 每100個問題顯示一次進度
            if (idx + 1) % 100 == 0:
                print(f"   已完成: {idx + 1}/{len(df)} ({(idx + 1)/len(df)*100:.1f}%)")
                
        except Exception as e:
            print(f"❌ 第{idx+1}行預測失敗: {e}")
            # 使用默認答案
            results.append({
                'ID': idx + 1,
                'Answer': 'A'
            })
    
    # 4. 保存結果（按照sample_submission.csv格式）
    print("💾 保存預測結果...")
    try:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('ID')  # 按ID排序
        results_df.to_csv(output_path, index=False)
        print(f"✅ 預測結果已保存至: {output_path}")
        
        # 保存詳細推理過程（用於分析和調試）
        detailed_output_path = output_path.replace('.csv', '_detailed.csv')
        detailed_df = pd.DataFrame(detailed_results)
        detailed_df = detailed_df.sort_values('ID')
        detailed_df.to_csv(detailed_output_path, index=False)
        print(f"✅ 詳細推理過程已保存至: {detailed_output_path}")
        
        # 顯示結果統計
        print("\n📊 預測結果統計:")
        answer_counts = results_df['Answer'].value_counts()
        for answer, count in answer_counts.items():
            percentage = count / len(results_df) * 100
            print(f"   {answer}: {count} 個 ({percentage:.1f}%)")
            
    except Exception as e:
        print(f"❌ 保存結果失敗: {e}")
        return
    
    print("=" * 70)  
    print("🎉 推理增強預測完成！")
    print("=" * 70)

def main():
    """主函數"""
    # 配置文件路徑
    test_csv_path = "C:/Users/NTHUILST/Ray/DL/data/test-check-v2.csv"
    model_path = "./chinese_llm_mcq_model_qwen2.5_7b_reasoning"
    output_path = "./submission_reasoning.csv"
    
    # 預測模式選擇
    prediction_mode = "reasoning"  # 可選: "reasoning", "simple", "mixed"
    
    # 檢查文件是否存在
    if not os.path.exists(test_csv_path):
        print(f"❌ 測試文件不存在: {test_csv_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 顯示配置信息
    print("🔧 配置信息:")
    print(f"   測試文件: {test_csv_path}")
    print(f"   模型路徑: {model_path}")
    print(f"   輸出文件: {output_path}")
    print(f"   預測模式: {prediction_mode}")
    print()
    
    # 執行預測
    start_time = time.time()
    predict_test_data_with_reasoning(test_csv_path, model_path, output_path, prediction_mode)
    end_time = time.time()
    
    print(f"⏱️  總耗時: {end_time - start_time:.1f} 秒")

if __name__ == "__main__":
    main()
