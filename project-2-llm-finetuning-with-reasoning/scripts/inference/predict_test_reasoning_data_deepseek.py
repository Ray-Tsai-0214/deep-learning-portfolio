"""
使用最佳DeepSeek Checkpoint進行預測
基於checkpoint-600 (最佳驗證loss: 0.6686)
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import csv
from tqdm import tqdm
import time
import re
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BestDeepSeekPredictor:
    def __init__(self, 
                 base_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                 checkpoint_path="./deepseek_reasoning_deepseek_r1_14b_mixed/checkpoint-600"):
        """
        使用最佳checkpoint初始化預測器
        
        Args:
            base_model_name: 基礎模型名稱
            checkpoint_path: 最佳checkpoint路徑
        """
        logger.info("🔄 加載最佳DeepSeek checkpoint...")
        logger.info(f"   基礎模型: {base_model_name}")
        logger.info(f"   Checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint不存在: {checkpoint_path}")
        
        try:
            # 1. 加載tokenizer
            logger.info("   📝 加載tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # 2. 加載基礎模型
            logger.info("   🤖 加載基礎模型...")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                load_in_4bit=True  # 節省顯存
            )
            
            # 3. 加載LoRA適配器
            logger.info("   🎯 加載LoRA適配器...")
            self.model = PeftModel.from_pretrained(
                self.base_model, 
                checkpoint_path,
                torch_dtype=torch.bfloat16
            )
            
            # 4. 設置為評估模式
            self.model.eval()
            
            logger.info("✅ 最佳DeepSeek模型加載完成!")
            
        except Exception as e:
            logger.error(f"❌ 模型加載失敗: {e}")
            raise e    
    def predict_single(self, question, option_a, option_b, option_c, option_d, mode="mixed"):
        """
        對單個問題進行預測
        
        Args:
            question: 問題文本
            option_a, option_b, option_c, option_d: 選項
            mode: 預測模式 ("simple", "reasoning", "mixed")
        """
        
        # 構建選項文本
        options = f"A. {option_a}\nB. {option_b}\nC. {option_c}\nD. {option_d}"
        
        # 根據訓練時的格式構建prompt
        if mode == "reasoning":
            system_message = "You are an analytical assistant. Provide brief reasoning before giving your answer to multiple choice questions."
            user_message = f"請分析以下選擇題並簡要說明推理過程，然後給出答案：\n\n問題：{question}\n\n選項：\n{options}"
        else:
            system_message = "You are a helpful assistant that provides objective answers to multiple choice questions. Always respond with the correct letter only."
            user_message = f"請回答以下選擇題，只需回答選項字母（A、B、C或D）：\n\n問題：{question}\n\n選項：\n{options}\n\n答案："
        
        # 構建DeepSeek格式的對話
        chat_text = f"System: {system_message}\n\nUser: {user_message}\n\nAssistant: "
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                chat_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.model.device)
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64 if mode == "simple" else 200,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解碼回答
            response = self.tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):], 
                skip_special_tokens=True
            )
            
            # 提取答案
            predicted_answer = self.extract_answer(response)
            
            return predicted_answer, response.strip()
            
        except Exception as e:
            logger.warning(f"⚠️ 預測失敗: {e}")
            return 'A', f"錯誤: {str(e)}"    
    def extract_answer(self, response):
        """強化版答案提取"""
        
        # 清理response
        response = response.strip()
        
        # 策略1: 查找單獨的A/B/C/D
        single_letter = re.search(r'^([ABCD])$', response)
        if single_letter:
            return single_letter.group(1)
        
        # 策略2: 查找開頭的字母
        first_letter = re.search(r'^([ABCD])', response)
        if first_letter:
            return first_letter.group(1)
        
        # 策略3: 查找答案模式
        answer_patterns = [
            r'答案[：:]\s*([ABCD])',
            r'最終答案[：:]\s*([ABCD])',
            r'選擇\s*([ABCD])',
            r'正確答案[：:]\s*([ABCD])',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)
        
        # 策略4: 查找任何A/B/C/D (取最後一個)
        letters = re.findall(r'([ABCD])', response)
        if letters:
            return letters[-1]
        
        # 默認返回A
        logger.warning(f"⚠️ 無法解析答案: '{response[:100]}...'，默認返回A")
        return 'A'

def predict_test_data_best_checkpoint(
    test_csv_path, 
    checkpoint_path="./deepseek_reasoning_deepseek_r1_14b_mixed/checkpoint-600",
    output_path="./submission_best_deepseek.csv",
    prediction_mode="mixed"
):
    """
    使用最佳checkpoint對測試數據進行批量預測
    
    Args:
        test_csv_path: 測試數據路徑
        checkpoint_path: 最佳checkpoint路徑
        output_path: 輸出文件路徑
        prediction_mode: 預測模式
    """
    
    print("=" * 70)
    print("🎯 使用最佳DeepSeek Checkpoint進行預測")
    print(f"📊 Checkpoint: {checkpoint_path}")
    print(f"🎪 預測模式: {prediction_mode}")
    print("=" * 70)    
    # 1. 檢查文件存在性
    if not os.path.exists(test_csv_path):
        print(f"❌ 測試文件不存在: {test_csv_path}")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return
    
    # 2. 加載測試數據
    print("📊 加載測試數據...")
    try:
        df = pd.read_csv(test_csv_path)
        print(f"✅ 成功加載 {len(df)} 個測試樣本")
        
        # 檢查數據格式
        required_cols = ['ID', 'Question', 'Option A', 'Option B', 'Option C', 'Option D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ 缺少必要欄位: {missing_cols}")
            print(f"可用欄位: {list(df.columns)}")
            return
            
    except Exception as e:
        print(f"❌ 加載測試數據失敗: {e}")
        return
    
    # 3. 初始化預測器
    try:
        predictor = BestDeepSeekPredictor(checkpoint_path=checkpoint_path)
    except Exception as e:
        print(f"❌ 預測器初始化失敗: {e}")
        return    
    # 4. 批量預測
    print(f"🔮 開始預測 {len(df)} 個問題...")
    results = []
    
    start_time = time.time()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="預測進度"):
        try:
            question_id = int(row['ID'])
            question = str(row['Question']).strip()
            option_a = str(row['Option A']).strip()
            option_b = str(row['Option B']).strip()
            option_c = str(row['Option C']).strip()
            option_d = str(row['Option D']).strip()
            
            # 進行預測
            predicted_answer, reasoning = predictor.predict_single(
                question, option_a, option_b, option_c, option_d, 
                mode=prediction_mode
            )
            
            results.append({
                'ID': question_id,
                'Answer': predicted_answer
            })
            
            # 每50個樣本顯示進度
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed
                eta = (len(df) - idx - 1) / rate
                print(f"   進度: {idx + 1}/{len(df)} ({rate:.1f} samples/sec, ETA: {eta/60:.1f}min)")
                
        except Exception as e:
            print(f"❌ 第{idx+1}行預測失敗: {e}")
            results.append({
                'ID': idx + 1,
                'Answer': 'A'
            })    
    # 5. 保存結果
    print("💾 保存預測結果...")
    try:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('ID')
        
        # 驗證答案格式
        valid_answers = {'A', 'B', 'C', 'D'}
        invalid_count = 0
        
        for idx, row in results_df.iterrows():
            if row['Answer'] not in valid_answers:
                print(f"⚠️ 修復無效答案: ID {row['ID']}, '{row['Answer']}' -> 'A'")
                results_df.at[idx, 'Answer'] = 'A'
                invalid_count += 1
        
        if invalid_count > 0:
            print(f"🔧 修復了 {invalid_count} 個無效答案")
        
        # 保存CSV
        results_df.to_csv(output_path, index=False)
        print(f"✅ 預測結果已保存至: {output_path}")
        
        # 顯示結果統計
        print("\n📊 預測結果統計:")
        answer_counts = results_df['Answer'].value_counts().sort_index()
        for answer, count in answer_counts.items():
            percentage = count / len(results_df) * 100
            print(f"   {answer}: {count} 個 ({percentage:.1f}%)")
        
        # 計算總耗時
        total_time = time.time() - start_time
        print(f"\n⏱️ 總耗時: {total_time/60:.1f} 分鐘")
        print(f"📈 預期Kaggle分數: 0.70-0.78 (基於最佳checkpoint)")
        
    except Exception as e:
        print(f"❌ 保存結果失敗: {e}")
        return
    
    print("=" * 70)
    print("🎉 最佳Checkpoint預測完成!")
    print("=" * 70)
def main():
    """主函數"""
    
    # 配置
    config = {
        "test_file": "C:/Users/NTHUILST/Ray/DL/data/test-check-v2.csv",
        "checkpoint": "./deepseek_reasoning_deepseek_r1_14b_mixed/checkpoint-600",  # 最佳checkpoint
        "output": "./submission_best_deepseek.csv",
        "mode": "mixed"  # mixed模式平衡速度和質量
    }
    
    print("🎯 使用最佳DeepSeek Checkpoint進行預測")
    print(f"配置: {config}")
    
    # 檢查checkpoint是否存在
    if not os.path.exists(config["checkpoint"]):
        print(f"❌ 最佳checkpoint不存在: {config['checkpoint']}")
        print("\n可選方案:")
        print("1. 使用 checkpoint-2400 (最新但可能過擬合)")
        print("2. 使用 checkpoint-1800")
        print("3. 重新訓練模型")
        return
    
    # 開始預測
    predict_test_data_best_checkpoint(
        test_csv_path=config["test_file"],
        checkpoint_path=config["checkpoint"],
        output_path=config["output"],
        prediction_mode=config["mode"]
    )

if __name__ == "__main__":
    main()