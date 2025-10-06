#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import pandas as pd
import json
import numpy as np
import logging
import time
import gc
import sys
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel, PeftConfig, get_peft_model, prepare_model_for_kbit_training, LoraConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日誌系統 - 加入彩色輸出
class ColoredFormatter(logging.Formatter):
    """自定義格式化器，支援彩色輸出"""
    COLORS = {
        'WARNING': '\033[93m',  # 黃色
        'ERROR': '\033[91m',    # 紅色
        'DEBUG': '\033[94m',    # 藍色
        'INFO': '\033[92m',     # 綠色
        'RESET': '\033[0m'      # 重置
    }

    def format(self, record):
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message

# 配置日誌系統
file_handler = logging.FileHandler("qwen_prediction.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

# 針對X答案和問題源頭的專門記錄器
x_logger = logging.getLogger("x_answers")
x_logger.setLevel(logging.WARNING)
x_formatter = ColoredFormatter('\033[93m[X答案]\033[0m %(message)s')
x_handler = logging.StreamHandler()
x_handler.setFormatter(x_formatter)
x_logger.addHandler(x_handler)

def convert_to_serializable(obj):
    """將對象轉換為可JSON序列化格式"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "item") and callable(getattr(obj, "item")):
        return obj.item()
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

class QwenPredictor:
    def __init__(self, 
                 model_path="./chinese_llm_mcq_model_qwen2.5_1m", 
                 base_model_id="Qwen/Qwen2.5-14B-Instruct-1M",
                 output_file="submission_qwen2.5-1M_chi.csv", 
                 batch_size=None):
        """
        初始化Qwen預測器
        
        參數:
            model_path: 微調模型路徑
            base_model_id: 基礎模型ID
            output_file: 輸出檔案名稱
            batch_size: 批次大小，若為None則自動確定
        """
        self.model_path = model_path
        self.base_model_id = base_model_id
        self.output_file = output_file
        self.checkpoint_file = "prediction_checkpoint_qwen.csv"
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loading_retries = 3  # 模型載入重試次數
        
        # 設置記憶體回收閾值
        self.gc_threshold = 15  # 每處理這麼多批次後執行一次記憶體回收
        
        # 顯示GPU資訊
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"使用GPU: {gpu_name}, 顯存: {gpu_mem:.1f} GB")
        
        # X答案統計
        self.x_answers_count = 0
        self.x_answers_reasons = {}
        
        # 創建日誌目錄
        os.makedirs("logs", exist_ok=True)

    def load_tokenizer(self):
        """載入分詞器"""
        logger.info(f"載入分詞器: {self.base_model_id}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_id, 
                trust_remote_code=True,
                padding_side="left"
            )
            
            # 確保tokenizer有pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("分詞器載入成功")
            return True
        except Exception as e:
            logger.error(f"分詞器載入失敗: {e}")
            return False
    
    def try_multiple_loading_strategies(self):
        """嘗試多種模型載入策略"""
        for attempt in range(self.loading_retries):
            logger.info(f"嘗試載入模型 (嘗試 {attempt+1}/{self.loading_retries})...")
            
            try:
                # 清理記憶體
                torch.cuda.empty_cache()
                gc.collect()
                
                # 使用不同策略
                if attempt == 0:
                    return self.load_model_strategy_1()  # 4-bit量化策略
                elif attempt == 1:
                    return self.load_model_strategy_2()  # 8-bit量化策略
                else:
                    return self.load_model_strategy_3()  # CPU+GPU混合策略
            except Exception as e:
                logger.error(f"載入策略 {attempt+1} 失敗: {e}")
                
                # 最後一次嘗試失敗後不再隱藏異常
                if attempt == self.loading_retries - 1:
                    raise e
        
        return False

    def load_model_strategy_1(self):
        """載入策略1: 使用4-bit量化"""
        logger.info("使用4-bit量化載入模型...")
        
        # 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # 載入基礎模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={0: "20GiB", "cpu": "32GiB"}  # 明確指定記憶體分配
        )
        
        # 載入微調權重
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
        return True

    def load_model_strategy_2(self):
        """載入策略2: 使用8-bit量化"""
        logger.info("使用8-bit量化載入模型...")
        
        # 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        # 載入基礎模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            max_memory={0: "18GiB", "cpu": "32GiB"}
        )
        
        # 載入微調權重
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()
        
        return True

    def load_model_strategy_3(self):
        """載入策略3: CPU+GPU混合策略"""
        logger.info("使用CPU+GPU混合策略載入模型...")
        
        # 載入基礎模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map="balanced",  # 平衡分配到CPU和GPU
            offload_folder="offload_folder",  # 啟用磁碟卸載
            offload_state_dict=True,  # 卸載狀態字典到CPU
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # 獲取微調配置
        peft_config = PeftConfig.from_pretrained(self.model_path)
        
        # 載入微調權重
        self.model = PeftModel.from_pretrained(base_model, self.model_path, config=peft_config)
        self.model.eval()
        
        return True

    def manual_load_model(self):
        """手動載入策略: 直接處理LoRA權重"""
        logger.info("嘗試手動載入LoRA權重...")
        
        # 載入基礎模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 準備模型進行kbit訓練
        base_model = prepare_model_for_kbit_training(base_model)
        
        # 構建LoRA配置
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                           "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # 獲取PEFT模型
        model = get_peft_model(base_model, lora_config)
        
        # 嘗試載入權重
        try:
            model.load_adapter(self.model_path)
            self.model = model
            logger.info("手動載入LoRA權重成功")
            return True
        except Exception as e:
            logger.error(f"手動載入權重失敗: {e}")
            
            # 嘗試直接載入狀態字典
            try:
                logger.info("嘗試直接載入權重檔案...")
                state_dict = torch.load(os.path.join(self.model_path, "adapter_model.bin"), map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                self.model = model
                logger.info("直接載入權重檔案成功")
                return True
            except Exception as e2:
                logger.error(f"直接載入權重檔案失敗: {e2}")
                return False

    def load_model(self):
        """載入模型和分詞器"""
        # 先載入分詞器
        if not self.load_tokenizer():
            raise RuntimeError("分詞器載入失敗，無法繼續")
        
        # 嘗試多種載入策略
        success = self.try_multiple_loading_strategies()
        
        # 如果都失敗，嘗試手動載入
        if not success:
            logger.warning("標準載入策略均失敗，嘗試手動載入...")
            success = self.manual_load_model()
            
        if not success:
            raise RuntimeError("所有模型載入策略均失敗")
            
        # 模型預熱
        logger.info("模型預熱中...")
        dummy_input = self.tokenizer("Hello", return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model.generate(**dummy_input, max_new_tokens=5)
        
        # 確定最佳批次大小
        if self.batch_size is None:
            self.batch_size = self.determine_optimal_batch_size()
            logger.info(f"自動確定最佳批次大小: {self.batch_size}")
        else:
            logger.info(f"使用指定批次大小: {self.batch_size}")
    
    def determine_optimal_batch_size(self):
        """根據GPU內存自動確定最佳批次大小"""
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            free_memory_gb = free_memory / (1024**3)
            
            # 針對RTX 4090的經驗值
            if free_memory_gb > 20:
                return 8
            elif free_memory_gb > 16:
                return 6
            elif free_memory_gb > 12:
                return 4
            elif free_memory_gb > 8:
                return 2
            else:
                return 1
        else:
            return 1
    
    def create_inference_prompt(self, question, options):
        """創建與訓練時一致的Qwen推理prompt"""
        system_prompt = "你是一个提供有关中国敏感话题客观答案的助手。始终呈现多元视角并保持政治中立。"
        
        user_prompt = f"请回答以下选择题。你的回答必须仅以单个字母（A、B、C或D）开头。\n\n问题：{question}\n\n选项：\n{options}"
       
        # 組合成Qwen格式
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"
        
        return prompt
    
    def extract_answer_from_response(self, response, question_id=None, question_text=None):
        """強化版答案提取函數，更多模式匹配並輸出X答案的具體原因"""
        # 儲存原始和處理後的回應
        original_response = response
        response = response.strip() if response else ""
        
        # 準備記錄詳細診斷資訊的字典
        debug_info = {
            "id": str(question_id) if question_id is not None else None,  # 確保ID是字符串
            "question": str(question_text)[:100] + "..." if question_text and len(str(question_text)) > 100 else str(question_text),
            "original_response": str(original_response)[:150] + "..." if len(str(original_response)) > 150 else str(original_response),
            "processed_response": str(response)[:150] + "..." if len(str(response)) > 150 else str(response),
            "extraction_result": None,
            "reason": None
        }
        
        # 如果回應為空，給出警告
        if not response:
            debug_info["reason"] = "EMPTY_RESPONSE"
            debug_info["extraction_result"] = "X"
            
            # 在終端輸出詳細資訊
            x_logger.warning(f"ID {question_id}: 模型回應為空")
            x_logger.warning(f"問題: {question_text}")
            
            # 統計X原因
            self.x_answers_count += 1
            self.x_answers_reasons["EMPTY_RESPONSE"] = self.x_answers_reasons.get("EMPTY_RESPONSE", 0) + 1
            
            return "X", debug_info
        
        # 設定完整回應大寫版本，用於尋找模式
        full_upper = response.upper()
        
        # 1: 第一個非空白字符檢查
        first_char = None
        for char in response:
            if not char.isspace():
                first_char = char.upper()
                break
        
        if first_char in "ABCD":
            debug_info["reason"] = "FIRST_CHAR"
            debug_info["extraction_result"] = first_char
            return first_char, debug_info
        
        # 2: 前段檢查常見格式
        early_text = response[:50].upper()
        for option in "ABCD":
            # 匹配各種常見格式
            patterns = [
                f"{option}.", f"{option}:", f"{option})", f"{option},",
                f" {option} ", f"^{option} ", f"OPTION {option}", 
                f"ANSWER {option}", f"CHOICE {option}", f"SELECT {option}"
            ]
            for pattern in patterns:
                if pattern in early_text:
                    debug_info["reason"] = f"EARLY_PATTERN"
                    debug_info["extraction_result"] = option
                    return option, debug_info
        
        # 3: 尋找明確的選項表述
        option_phrases = [
            # 英文選項
            ("A", ["OPTION A", "ANSWER A", "ANSWER IS A", "CHOOSE A", "I CHOOSE A", "SELECTING A", "A IS CORRECT"]),
            ("B", ["OPTION B", "ANSWER B", "ANSWER IS B", "CHOOSE B", "I CHOOSE B", "SELECTING B", "B IS CORRECT"]),
            ("C", ["OPTION C", "ANSWER C", "ANSWER IS C", "CHOOSE C", "I CHOOSE C", "SELECTING C", "C IS CORRECT"]),
            ("D", ["OPTION D", "ANSWER D", "ANSWER IS D", "CHOOSE D", "I CHOOSE D", "SELECTING D", "D IS CORRECT"])
        ]
        
        for option, phrases in option_phrases:
            for phrase in phrases:
                if phrase in full_upper:
                    debug_info["reason"] = f"PHRASE_MATCH"
                    debug_info["extraction_result"] = option
                    return option, debug_info
        
        # 4: 使用正則表達式進行更強大的模式匹配
        regex_patterns = [
            # "I/The answer is X" 格式
            (r"\b(?:THE|MY|I|THIS)?\s*(?:ANSWER|CHOICE|OPTION|SELECTION)\s*(?:IS|WOULD BE|SHOULD BE)\s*(?:OPTION)?\s*([A-D])\b", "REGEX_IS_PATTERN"),
            # "I/We select/choose/pick X" 格式
            (r"\b(?:I|WE)\s*(?:SELECT|CHOOSE|PICK|OPT FOR)\s*(?:OPTION)?\s*([A-D])\b", "REGEX_CHOOSE_PATTERN"),
            # 其他常見表達式
            (r"\b(?:OPTION|ANSWER|CHOICE)\s*([A-D])\s*(?:IS CORRECT|IS RIGHT|IS ACCURATE)\b", "REGEX_CORRECT_PATTERN"),
            # 數字對應格式
            (r"\b(?:FIRST|1ST|ONE|1)\b.*\bOPTION\b", "NUMBER_FIRST"),
            (r"\b(?:SECOND|2ND|TWO|2)\b.*\bOPTION\b", "NUMBER_SECOND"),
            (r"\b(?:THIRD|3RD|THREE|3)\b.*\bOPTION\b", "NUMBER_THIRD"),
            (r"\b(?:FOURTH|4TH|FOUR|4)\b.*\bOPTION\b", "NUMBER_FOURTH")
        ]
        
        for pattern, reason_code in regex_patterns:
            match = re.search(pattern, full_upper)
            if match:
                if reason_code == "NUMBER_FIRST":
                    debug_info["reason"] = reason_code
                    debug_info["extraction_result"] = "A"
                    return "A", debug_info
                elif reason_code == "NUMBER_SECOND":
                    debug_info["reason"] = reason_code
                    debug_info["extraction_result"] = "B"
                    return "B", debug_info
                elif reason_code == "NUMBER_THIRD":
                    debug_info["reason"] = reason_code
                    debug_info["extraction_result"] = "C"
                    return "C", debug_info
                elif reason_code == "NUMBER_FOURTH":
                    debug_info["reason"] = reason_code
                    debug_info["extraction_result"] = "D"
                    return "D", debug_info
                elif match.group(1) in "ABCD":
                    option = match.group(1)
                    debug_info["reason"] = reason_code
                    debug_info["extraction_result"] = option
                    return option, debug_info
        
        # 5: 作為最後嘗試，尋找任何ABCD字母，返回第一個找到的
        for letter in "ABCD":
            if letter in full_upper:
                debug_info["reason"] = "ANY_LETTER_MATCH"
                debug_info["extraction_result"] = letter
                return letter, debug_info
        
        # 如果還沒找到，檢查文本是否暗示某個選項
        hint_words = [
            ("FIRST", "A"), ("1ST", "A"), ("ONE", "A"), 
            ("SECOND", "B"), ("2ND", "B"), ("TWO", "B"),
            ("THIRD", "C"), ("3RD", "C"), ("THREE", "C"),
            ("FOURTH", "D"), ("4TH", "D"), ("FOUR", "D")
        ]
        
        for hint, option in hint_words:
            if hint in full_upper:
                debug_info["reason"] = f"POSITION_HINT"
                debug_info["extraction_result"] = option
                return option, debug_info
        
        # 完全無法找到任何線索的情況 - 標記為X並輸出詳細資訊
        debug_info["reason"] = "NO_OPTION_FOUND"
        debug_info["extraction_result"] = "X"
        
        # 在終端輸出詳細資訊
        x_logger.warning(f"ID {question_id}: 無法從回應中提取選項")
        x_logger.warning(f"問題: {question_text}")
        x_logger.warning(f"模型回應: {response[:100]}...")
        
        # 顯示回應中是否包含ABCD任何字母（可能是格式不符合預期）
        contains_letters = []
        for letter in "ABCD":
            if letter in full_upper:
                position = full_upper.find(letter)
                context = full_upper[max(0, position-10):min(len(full_upper), position+10)]
                contains_letters.append(f"{letter}(位置{position}): ...{context}...")
        
        if contains_letters:
            x_logger.warning(f"回應中包含以下字母，但格式不符合提取條件:")
            for info in contains_letters:
                x_logger.warning(f"  {info}")
        else:
            x_logger.warning("回應中不包含任何ABCD字母")
        
        # 統計X原因
        self.x_answers_count += 1
        self.x_answers_reasons["NO_OPTION_FOUND"] = self.x_answers_reasons.get("NO_OPTION_FOUND", 0) + 1
        
        return "X", debug_info
    
    def process_batch(self, batch_rows, batch_ids):
        """處理一批問題並返回答案"""
        try:
            # 準備批量輸入
            batch_prompts = []
            batch_questions = []  # 保存問題內容用於診斷
            for row in batch_rows:
                question = row['question']
                batch_questions.append(question)
                options = f"A. {row['option_A']}\nB. {row['option_B']}\nC. {row['option_C']}\nD. {row['option_D']}"
                prompt = self.create_inference_prompt(question, options)
                batch_prompts.append(prompt)
            
            # 批量編碼
            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.device)
            
            # 為RTX 4090優化的生成配置
            generation_config = GenerationConfig(
                max_new_tokens=48,         # 增加長度，提高回答完整性
                do_sample=False,           # 確定性輸出
                temperature=0.1,           # 低溫度增加確定性
                top_p=0.95,                # 控制詞彙分佈
                repetition_penalty=1.1,    # 輕微懲罰重複
                num_beams=1,               # 貪婪解碼
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # 解碼響應
            batch_results = []
            x_responses = []  # 專門記錄導致X的回應，便於診斷
            
            for i, output_ids in enumerate(outputs):
                response_ids = output_ids[inputs.input_ids.shape[1]:]
                response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                
                # 提取答案並記錄調試信息
                question_id = batch_ids[i]
                question_text = batch_questions[i]
                answer, debug_info = self.extract_answer_from_response(response, question_id, question_text)
                batch_results.append({"ID": question_id, "answer": answer})
                
                # 如果是X答案，記錄更詳細資訊
                if answer == "X":
                    x_responses.append({
                        "id": str(question_id),
                        "question": str(question_text),
                        "response": str(response),
                        "reason": debug_info.get("reason", "未知原因")
                    })
            
            # 保存X答案調試信息到文件
            if x_responses:
                try:
                    x_log_file = os.path.join("logs", "x_answers.txt")
                    with open(x_log_file, "a", encoding="utf-8") as f:
                        for item in x_responses:
                            f.write(f"ID: {item['id']}\n")
                            f.write(f"問題: {item['question']}\n")
                            f.write(f"回應: {item['response'][:200]}...\n")
                            f.write(f"原因: {item['reason']}\n")
                            f.write("-" * 50 + "\n")
                except Exception as e:
                    logger.error(f"保存X答案日誌時出錯: {e}")
            
            return batch_results
            
        except Exception as e:
            logger.error(f"批處理錯誤: {e}")
            # 錯誤時返回A答案，避免產生X
            error_results = [{"ID": id_value, "answer": "X"} for id_value in batch_ids]
            
            # 記錄這些錯誤
            logger.error(f"處理批次 {batch_ids} 時發生錯誤，使用默認答案A")
            
            return error_results

    def generate_predictions(self, test_file):
        """生成預測結果"""
        # 載入測試數據
        test_df = pd.read_csv(test_file)
        logger.info(f"測試文件包含 {len(test_df)} 個問題")
        
        # 確定ID列名
        id_col = "ID" if "ID" in test_df.columns else test_df.columns[0]
        
        # 檢查是否有檢查點文件
        processed_ids = set()
        results = []
        
        if os.path.exists(self.checkpoint_file):
            checkpoint_df = pd.read_csv(self.checkpoint_file)
            processed_ids = set(checkpoint_df[id_col].astype(str))
            results = checkpoint_df.to_dict('records')
            logger.info(f"從檢查點恢復，已處理 {len(processed_ids)} 個問題")
        
        # 過濾掉已處理的問題
        remaining_indices = [
            i for i, row in test_df.iterrows() 
            if str(row[id_col]) not in processed_ids
        ]
        
        logger.info(f"剩餘 {len(remaining_indices)} 個問題需要處理")
        
        # 計時
        start_time = time.time()
        last_checkpoint_time = start_time
        
        # 批處理問題
        total_batches = (len(remaining_indices)-1) // self.batch_size + 1 if len(remaining_indices) > 0 else 0
        
        for batch_index in range(0, len(remaining_indices), self.batch_size):
            batch_start = batch_index
            batch_end = min(batch_start + self.batch_size, len(remaining_indices))
            batch_indices = remaining_indices[batch_start:batch_end]
            batch_rows = [test_df.iloc[i] for i in batch_indices]
            batch_ids = [row[id_col] for row in batch_rows]
            
            # 處理批次
            current_batch = batch_index//self.batch_size + 1
            logger.info(f"處理批次 {current_batch}/{total_batches} ({current_batch/total_batches*100:.1f}%)")
            batch_results = self.process_batch(batch_rows, batch_ids)
            results.extend(batch_results)
            
            # 定期保存檢查點
            current_time = time.time()
            if (current_time - last_checkpoint_time > 120) or (batch_end >= len(remaining_indices)):
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_csv(self.checkpoint_file, index=False)
                last_checkpoint_time = current_time
                
                # 顯示進度信息
                elapsed_time = max(0.001, current_time - start_time)  # 避免除零錯誤
                total_processed = len(results)
                questions_per_second = total_processed / elapsed_time
                remaining_questions = len(test_df) - total_processed
                estimated_time_left = remaining_questions / questions_per_second if questions_per_second > 0 else 0
                
                logger.info(f"進度: {total_processed}/{len(test_df)} 題 ({total_processed/len(test_df)*100:.1f}%)")
                logger.info(f"速度: {questions_per_second:.2f} 題/秒")
                logger.info(f"已用時間: {elapsed_time/60:.1f}分鐘, 預計剩餘: {estimated_time_left/60:.1f}分鐘")
                
                # 顯示X答案統計
                if self.x_answers_count > 0:
                    x_percent = (self.x_answers_count / total_processed) * 100
                    logger.warning(f"目前有 {self.x_answers_count} 個X答案 ({x_percent:.1f}%)")
                    logger.warning("X答案原因分類:")
                    for reason, count in sorted(self.x_answers_reasons.items(), key=lambda x: x[1], reverse=True):
                        reason_percent = (count / self.x_answers_count) * 100
                        logger.warning(f"  {reason}: {count} 個 ({reason_percent:.1f}%)")
            
            # 定期執行垃圾回收
            if batch_index > 0 and batch_index % (self.gc_threshold * self.batch_size) == 0:
                logger.info("執行內存回收...")
                torch.cuda.empty_cache()
                gc.collect()

                # 顯示GPU使用情況
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    reserved = torch.cuda.memory_reserved() / (1024**3)
                    logger.info(f"GPU記憶體: 已分配 {allocated:.2f}GB, 已預留 {reserved:.2f}GB")
        
        # 生成最終CSV
        final_df = pd.DataFrame(results)
        
        # 確保ID列格式正確
        try:
            final_df[id_col] = final_df[id_col].astype(int)
        except:
            pass
        
        # 按ID排序
        final_df = final_df.sort_values(id_col)
        
        # 檢查是否有缺失的ID並填充
        all_ids = set(test_df[id_col].astype(str))
        result_ids = set(final_df[id_col].astype(str))
        missing_ids = all_ids - result_ids
        
        if missing_ids:
            logger.warning(f"發現 {len(missing_ids)} 個缺失的ID，使用默認答案A")
            missing_rows = []
            for missing_id in missing_ids:
                missing_rows.append({id_col: missing_id, "answer": "X"})
                
            if missing_rows:
                missing_df = pd.DataFrame(missing_rows)
                final_df = pd.concat([final_df, missing_df], ignore_index=True)
        
        # 保存最終結果
        final_df.to_csv(self.output_file, index=False)
        
        # 統計與報告
        elapsed_time = max(0.001, time.time() - start_time)  # 避免除零錯誤
        self.report_statistics(final_df, elapsed_time)
        
        
        
        return final_df
    
    def report_statistics(self, final_df, elapsed_time):
        """報告預測結果統計"""
        total_count = len(final_df)
        
        # 答案分佈
        answer_counts = final_df["answer"].value_counts()
        
        logger.info("\n" + "="*50)
        logger.info(f"預測完成，結果已保存至 {self.output_file}")
        logger.info(f"總處理時間: {elapsed_time/60:.2f}分鐘")
        logger.info(f"平均速度: {total_count/elapsed_time:.2f} 題/秒")
        logger.info("\n答案分佈:")
        
        for answer, count in answer_counts.items():
            logger.info(f"{answer}: {count} 題 ({count/total_count*100:.1f}%)")
        
        logger.info("="*50)
 

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="使用微調的Qwen2.5模型生成預測")
    parser.add_argument("--model_path", type=str, default="./chinese_llm_mcq_model_qwen2.5_1m", help="微調模型路徑")
    parser.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-14B-Instruct-1M", help="基礎模型ID")
    parser.add_argument("--test_file", type=str, default="./data/test-v2.csv", help="測試文件路徑")
    parser.add_argument("--output_file", type=str, default="submission_qwen2.5_1m_chi.csv", help="輸出文件名")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小，不指定則自動決定")
    
    args = parser.parse_args()
    
    # 顯示主要參數
    logger.info(f"微調模型: {args.model_path}")
    logger.info(f"基礎模型: {args.base_model_id}")
    logger.info(f"測試文件: {args.test_file}")
    
    # 記錄 PyTorch 和 CUDA 版本
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
    
    # 設置環境變數以優化性能
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    try:
        # 創建預測器實例
        predictor = QwenPredictor(
            model_path=args.model_path,
            base_model_id=args.base_model_id,
            output_file=args.output_file,
            batch_size=args.batch_size
        )
        
        # 載入模型
        predictor.load_model()
        
        # 生成預測
        predictor.generate_predictions(args.test_file)
        
    except Exception as e:
        logger.exception(f"程序執行過程中發生錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
