#!/usr/bin/env python3
"""
GRPO訓練腳本 - 中文推理數據50%子集
專門用於Kaggle #3: GRPO with Reasoning訓練

主要特色:
- Group Relative Policy Optimization (GRPO)
- 中文敏感議題推理能力訓練
- 中立性感知的獎勵函數
- 40小時長時間穩定訓練
"""
#!/usr/bin/env python3
"""
完全修復版GRPO訓練腳本 - 解決multiprocessing pickle問題
Complete Fixed GRPO Training Script for Chinese LLM Reasoning
"""

import os
import sys
import json
import yaml
import logging
import torch
import random
import pandas as pd
import numpy as np
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import traceback

# 核心ML庫
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model

# 監控和記錄
import wandb

# 中文模型映射
CHINESE_MODELS = {
    "qwen2.5_7b": "Qwen/Qwen2.5-7B-Instruct",
    "chatglm3_6b": "THUDM/chatglm3-6b",
    "baichuan2_7b": "baichuan-inc/Baichuan2-7B-Chat",
    "internlm2_7b": "internlm/internlm2-chat-7b"
}

# 全局定義 data collator 以支持多進程
def global_data_collator(features):
    """全局數據收集器，支持多進程序列化"""
    # 這是一個簡單的實現，可以根據需要自定義
    return features

# 設置日誌
def setup_logging():
    """設置日誌系統"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建logs目錄
    os.makedirs("/home/ubuntu/DL/kaggle#3/logs", exist_ok=True)
    
    # 配置主日誌
    main_log_file = f"/home/ubuntu/DL/kaggle#3/logs/grpo_training_{timestamp}.log"
    progress_log_file = f"/home/ubuntu/DL/kaggle#3/logs/training_progress_{timestamp}.log"
    error_log_file = f"/home/ubuntu/DL/kaggle#3/logs/training_errors_{timestamp}.log"
    
    # 配置根日誌器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # 添加進度日誌處理器
    progress_handler = logging.FileHandler(progress_log_file)
    progress_handler.setLevel(logging.INFO)
    logger.addHandler(progress_handler)
    
    # 添加錯誤日誌處理器
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    logger.addHandler(error_handler)
    
    # 返回日誌文件路徑
    return logger, main_log_file, progress_log_file, error_log_file

def load_config(config_path: str) -> Dict:
    """載入配置文件 - 增加類型修復"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"📖 Loading configuration from: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 修復數據類型 - 確保關鍵參數是正確的類型
        if 'training' in config:
            training_config = config['training']
            
            # 修復學習率類型
            if 'learning_rate' in training_config:
                lr = training_config['learning_rate']
                if isinstance(lr, str):
                    try:
                        training_config['learning_rate'] = float(lr)
                        logger.info(f"🔧 Fixed learning_rate type: {lr} -> {training_config['learning_rate']}")
                    except ValueError:
                        logger.warning(f"⚠️ Could not convert learning_rate '{lr}' to float, using default 5e-5")
                        training_config['learning_rate'] = 5e-5
            
            # 修復其他可能的類型問題
            int_params = ['batch_size', 'gradient_accumulation_steps', 'num_epochs', 'warmup_steps', 
                         'logging_steps', 'save_steps', 'eval_steps']
            for param in int_params:
                if param in training_config and isinstance(training_config[param], str):
                    try:
                        training_config[param] = int(training_config[param])
                        logger.info(f"🔧 Fixed {param} type: str -> int")
                    except ValueError:
                        logger.warning(f"⚠️ Could not convert {param} to int")
            
            # 修復浮點數參數
            if 'data' in config:
                float_params = ['train_test_split']
                for param in float_params:
                    if param in config['data'] and isinstance(config['data'][param], str):
                        try:
                            config['data'][param] = float(config['data'][param])
                            logger.info(f"🔧 Fixed {param} type: str -> float")
                        except ValueError:
                            logger.warning(f"⚠️ Could not convert {param} to float")
        
        # 強制啟用快速測試模式
        if 'training' not in config:
            config['training'] = {}
        
        config['training']['quick_test'] = True
        config['training']['sample_ratio'] = 0.5  # 只用1%數據快速測試
        
        # 【關鍵修復】強制設置 num_workers 為 0 以避免 pickle 問題
        if 'system' not in config:
            config['system'] = {}
        config['system']['num_workers'] = 0
        
        logger.info("✅ Configuration loaded and fixed successfully")
        logger.info(f"🔍 配置的模型: {config.get('model', {}).get('name', 'unknown')}")
        logger.info(f"🔍 批次大小: {config.get('training', {}).get('batch_size', 'unknown')}")
        logger.info(f"🔍 學習率: {config.get('training', {}).get('learning_rate', 'unknown')}")
        logger.info(f"🔍 最大長度: {config.get('data', {}).get('max_length', 'unknown')}")
        logger.info(f"🚀 快速測試模式: {config.get('training', {}).get('quick_test', False)}")
        logger.info(f"🔧 數據加載器工作進程: {config.get('system', {}).get('num_workers', 0)}")
        
        return config
        
    except Exception as e:
        logger.error(f"❌ 配置載入失敗: {e}")
        raise

def get_gpu_memory_info():
    """獲取GPU記憶體信息"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(device).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(device) / 1e9
        return f"{allocated:.1f}GB/{total:.1f}GB"
    return "N/A"

def setup_environment():
    """設置訓練環境"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔧 Setting up training environment...")
    
    # 設置隨機種子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        cuda_version = torch.version.cuda
        pytorch_version = torch.__version__
        
        logger.info(f"🔥 GPU: {gpu_name}")
        logger.info(f"🔥 CUDA版本: {cuda_version}")
        logger.info(f"🔥 GPU記憶體: {gpu_memory:.2f} GB")
        logger.info(f"🐍 PyTorch版本: {pytorch_version}")
    else:
        logger.warning("⚠️  CUDA不可用，將使用CPU訓練")

class ChineseReasoningDataProcessor:
    """中文推理數據處理器"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.processed_data = None
        self.logger = logging.getLogger(__name__)
        
    def load_data(self):
        """載入數據"""
        try:
            self.logger.info(f"📊 Loading data from: {self.data_path}")
            
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"數據文件不存在: {self.data_path}")
            
            self.df = pd.read_csv(self.data_path, sep='\t')
            
            self.logger.info(f"✅ Loaded {len(self.df)} samples")
            self.logger.info(f"📋 Columns: {list(self.df.columns)}")
            
            # 統計信息
            self.logger.info("📈 Data Statistics:")
            self.logger.info(f"   - Total rows: {len(self.df)}")
            self.logger.info(f"   - Missing values: {self.df.isnull().sum().sum()}")
            
            file_size = os.path.getsize(self.data_path) / (1024 * 1024)
            self.logger.info(f"   - File size: {file_size:.2f} MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 數據載入失敗: {e}")
            raise
    
    def extract_reasoning_components(self, reasoning_text: str) -> Dict[str, str]:
        """提取推理組件"""
        try:
            components = {
                "analysis": "分析步驟",
                "key_point": "關鍵要點", 
                "conclusion": "結論"
            }
            
            # 簡化處理：將推理文本分段
            text = str(reasoning_text).strip()
            sentences = text.split('。')
            
            if len(sentences) >= 3:
                components["analysis"] = sentences[0] + "。"
                components["key_point"] = sentences[1] + "。"
                components["conclusion"] = sentences[-1] if sentences[-1] else sentences[-2]
            else:
                components["analysis"] = text
                components["key_point"] = text
                components["conclusion"] = text
            
            return components
            
        except Exception as e:
            self.logger.warning(f"推理組件提取失敗: {e}")
            return {
                "analysis": str(reasoning_text),
                "key_point": str(reasoning_text),
                "conclusion": str(reasoning_text)
            }
    
    def create_grpo_pairs(self, question: str, options: str, correct_answer: str, 
                         reasoning_components: Dict[str, str]) -> List[Dict]:
        """創建GRPO偏好對"""
        pairs = []
        
        try:
            # 基本提示
            prompt = f"問題：{question}\n選項：\n{options}\n請選擇正確答案並說明理由。"
            
            # 正確回答（chosen）
            chosen = f"答案：{correct_answer}\n理由：{reasoning_components['analysis']}"
            
            # 創建錯誤回答（rejected）
            wrong_answers = ['A', 'B', 'C', 'D']
            if correct_answer in wrong_answers:
                wrong_answers.remove(correct_answer)
            
            for wrong_answer in wrong_answers[:2]:  # 只取前2個錯誤答案
                rejected = f"答案：{wrong_answer}\n理由：這個選項不正確。"
                
                pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
            
            return pairs
            
        except Exception as e:
            self.logger.warning(f"創建偏好對失敗: {e}")
            return []
    
    def process_data_for_grpo(self) -> List[Dict]:
        """處理數據用於GRPO訓練"""
        if self.df is None:
            self.load_data()
        
        self.logger.info("🔄 Processing data for GRPO training...")
        
        all_pairs = []
        failed_count = 0
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing data"):
            try:
                question = str(row['question']).strip()
                options = f"A. {row['option_A']}\nB. {row['option_B']}\nC. {row['option_C']}\nD. {row['option_D']}"
                correct_answer = str(row['correct_answer']).strip()
                
                # 提取推理組件
                reasoning_components = self.extract_reasoning_components(str(row['reasoning_answer']))
                
                # 創建偏好對
                pairs = self.create_grpo_pairs(question, options, correct_answer, reasoning_components)
                all_pairs.extend(pairs)
                
                # 進度報告
                if (idx + 1) % 1000 == 0:
                    self.logger.info(f"📊 Processed {idx + 1}/{len(self.df)} rows, generated {len(all_pairs)} pairs")
                
            except Exception as e:
                self.logger.warning(f"Error processing row {idx}: {e}")
                failed_count += 1
                continue
        
        self.logger.info("✅ Data processing completed!")
        self.logger.info("📊 Statistics:")
        self.logger.info(f"   - Total samples processed: {len(self.df)}")
        self.logger.info(f"   - Failed rows: {failed_count}")
        self.logger.info(f"   - Generated preference pairs: {len(all_pairs)}")
        self.logger.info(f"   - Average pairs per sample: {len(all_pairs)/len(self.df):.2f}")
        
        self.processed_data = all_pairs
        return all_pairs

class QuickTestDataProcessor(ChineseReasoningDataProcessor):
    """支持快速測試的數據處理器"""
    
    def __init__(self, data_path: str, quick_test: bool = False, sample_ratio: float = 0.5):
        super().__init__(data_path)
        self.quick_test = quick_test
        self.sample_ratio = sample_ratio
    
    def process_data_for_grpo(self) -> List[Dict]:
        """處理數據用於GRPO訓練 - 支持快速測試"""
        if self.df is None:
            self.load_data()
        
        # 快速測試模式
        if self.quick_test:
            original_size = len(self.df)
            sample_size = max(1, int(original_size * self.sample_ratio))
            self.df = self.df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            self.logger.info(f"🚀 Quick test mode: using {sample_size}/{original_size} samples ({self.sample_ratio*100:.1f}%)")
        
        self.logger.info("🔄 Processing data for GRPO training...")
        
        all_pairs = []
        failed_count = 0
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing data"):
            try:
                question = str(row['question']).strip()
                options = f"A. {row['option_A']}\nB. {row['option_B']}\nC. {row['option_C']}\nD. {row['option_D']}"
                correct_answer = str(row['correct_answer']).strip()
                
                # 簡化推理處理
                reasoning_text = str(row['reasoning_answer'])
                # 移除XML標籤並截短
                clean_reasoning = re.sub(r'<[^>]+>', '', reasoning_text)
                clean_reasoning = ' '.join(clean_reasoning.split())[:300] + "..."
                
                reasoning_components = {
                    "analysis": clean_reasoning,
                    "key_point": "基於分析",
                    "conclusion": "得出答案"
                }
                
                # 創建偏好對 - 只創建1個，減少數據量
                pairs = self.create_simplified_grpo_pairs(question, options, correct_answer, reasoning_components)
                all_pairs.extend(pairs)
                
            except Exception as e:
                self.logger.warning(f"Error processing row {idx}: {e}")
                failed_count += 1
                continue
        
        self.logger.info("✅ Data processing completed!")
        self.logger.info("📊 Statistics:")
        self.logger.info(f"   - Total samples processed: {len(self.df)}")
        self.logger.info(f"   - Failed rows: {failed_count}")
        self.logger.info(f"   - Generated preference pairs: {len(all_pairs)}")
        
        self.processed_data = all_pairs
        return all_pairs
    
    def create_simplified_grpo_pairs(self, question: str, options: str, correct_answer: str, reasoning_components: Dict[str, str]) -> List[Dict]:
        """創建簡化的GRPO偏好對"""
        pairs = []
        
        try:
            # 簡化的提示格式
            prompt = f"Question: {question}\nOptions:\n{options}\nAnswer:"
            
            # 正確回答（chosen）
            chosen = f"{correct_answer}\nReasoning: {reasoning_components['analysis']}"
            
            # 錯誤回答（rejected） - 只創建1個
            wrong_answers = ['A', 'B', 'C', 'D']
            if correct_answer in wrong_answers:
                wrong_answers.remove(correct_answer)
            
            wrong_answer = wrong_answers[0]
            rejected = f"{wrong_answer}\nReasoning: This option is incorrect."
            
            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
            
            return pairs
            
        except Exception as e:
            self.logger.warning(f"創建偏好對失敗: {e}")
            return []

class ChineseLLMSetup:
    """中文大語言模型設置類"""
    
    def __init__(self, model_name: str, use_4bit: bool = True):
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.model_id = CHINESE_MODELS.get(model_name, model_name)
        self.model = None
        self.tokenizer = None
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🤖 初始化模型設置: {model_name}")
        self.logger.info(f"🔍 模型ID: {self.model_id}")
        self.logger.info(f"🔧 使用4bit量化: {use_4bit}")
    
    def setup_quantization(self):
        """設置量化配置"""
        if not self.use_4bit:
            return None
            
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.logger.info("✅ 4bit量化配置已設置")
        return quantization_config
    
    def load_model_and_tokenizer(self):
        """載入模型和tokenizer"""
        try:
            self.logger.info("📝 Loading tokenizer...")
            
            # 載入tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.logger.info("✅ Tokenizer loaded successfully!")
            
            # 載入模型
            self.logger.info("🧠 Loading model...")
            quantization_config = self.setup_quantization()
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    attn_implementation="flash_attention_2"
                )
                self.logger.info("✅ Flash Attention 2 已啟用")
            except Exception as e:
                self.logger.warning(f"Flash attention failed, falling back: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            
            self.logger.info("✅ Model loaded successfully!")
            
            # 顯示記憶體使用
            if torch.cuda.is_available():
                memory_info = get_gpu_memory_info()
                self.logger.info(f"🔥 GPU記憶體使用: {memory_info}")
            
            return self.model, self.tokenizer
            
        except Exception as e:
            self.logger.error(f"❌ 模型載入失敗: {e}")
            raise
    
    def setup_lora(self, config: Dict):
        """設置LoRA配置"""
        try:
            self.logger.info("🔧 Setting up LoRA configuration...")
            
            # 根據模型類型選擇target_modules
            if "qwen" in self.model_id.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "chatglm" in self.model_id.lower():
                target_modules = ["query_key_value", "dense"]
            else:
                target_modules = ["q_proj", "v_proj"]
            
            lora_config = LoraConfig(
                r=config['lora']['r'],
                lora_alpha=config['lora']['lora_alpha'],
                target_modules=target_modules,
                lora_dropout=config['lora']['lora_dropout'],
                bias=config['lora']['bias'],
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            
            # 顯示可訓練參數
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            
            self.logger.info(f"📊 Trainable parameters: {trainable_params:,}")
            self.logger.info(f"📊 All parameters: {all_params:,}")
            self.logger.info(f"📊 Trainable ratio: {100 * trainable_params / all_params:.2f}%")
            
            return self.model
            
        except Exception as e:
            self.logger.error(f"❌ LoRA設置失敗: {e}")
            raise

def setup_wandb(config: Dict):
    """設置Weights & Biases記錄"""
    logger = logging.getLogger(__name__)
    
    if config['system']['use_wandb']:
        try:
            logger.info("📊 Setting up WandB...")
            
            run_name = f"grpo-training-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            wandb.init(
                project=config['system']['wandb_project'],
                name=run_name,
                config=config
            )
            logger.info("✅ WandB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")

# 全局定義reward function 以支持多進程
def global_reward_function(prompts, completions, **kwargs):
    """Global GRPO reward function with correct signature"""
    import torch
    
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # 簡單獎勵邏輯
        if "Answer:" in completion or "答案：" in completion:
            reward = 1.0
        else:
            reward = 0.5
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)

def main():
    """主訓練函數 - 修復版本"""
    
    print("="*60)
    print("🚀 FIXED GRPO Training for Chinese LLM Reasoning (v2)")
    print("="*60)
    
    # 設置日誌
    logger, main_log_file, progress_log_file, error_log_file = setup_logging()
    
    try:
        logger.info("🎯 Starting FIXED GRPO training session")
        logger.info(f"📅 Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 載入配置（使用修復後的載入器）
        logger.info("📖 Loading configuration...")
        config = load_config('/home/ubuntu/DL/kaggle#3/configs/training_config_fixed.yaml')
        
        # 設置環境
        logger.info("🔧 Setting up environment...")
        setup_environment()
        
        # 設置WandB
        setup_wandb(config)
        
        logger.info("="*40)
        logger.info("📊 STEP 1: Processing training data...")
        
        # 步驟1: 使用快速測試數據處理器
        data_processor = QuickTestDataProcessor(
            data_path=config['data']['train_path'],
            quick_test=config['training'].get('quick_test', True),
            sample_ratio=config['training'].get('sample_ratio', 0.5)
        )
        preference_pairs = data_processor.process_data_for_grpo()
        
        if not preference_pairs:
            raise ValueError("沒有生成任何訓練數據對")
        
        logger.info(f"📊 Generated {len(preference_pairs)} preference pairs for quick test")
        
        logger.info("="*40)
        logger.info("🤖 STEP 2: Setting up model...")
        
        # 步驟2: 設置模型
        model_setup = ChineseLLMSetup(
            model_name=config['model']['name'],
            use_4bit=config['model']['use_4bit_quantization']
        )
        model, tokenizer = model_setup.load_model_and_tokenizer()
        model = model_setup.setup_lora(config)
        
        logger.info("="*40)
        logger.info("📝 STEP 3: Preparing dataset...")
        
        # 步驟3: 準備數據集 - 改進版tokenization
        def improved_tokenize_function(examples):
            """改進的tokenize函數，正確處理GRPO格式"""
            max_length = config['data']['max_length']
            
            # 確保所有輸入都是字符串
            prompts = [str(p) for p in examples["prompt"]]
            chosen = [str(c) for c in examples["chosen"]]
            rejected = [str(r) for r in examples["rejected"]]
            
            # Tokenize prompts
            prompt_encodings = tokenizer(
                prompts, 
                max_length=max_length,
                truncation=True,
                padding=False
            )
            
            # Tokenize chosen responses
            chosen_encodings = tokenizer(
                chosen,
                max_length=max_length,
                truncation=True,
                padding=False
            )
            
            # Tokenize rejected responses
            rejected_encodings = tokenizer(
                rejected,
                max_length=max_length,
                truncation=True,
                padding=False
            )
            
            return {
                "prompt": prompts,
                "chosen": chosen, 
                "rejected": rejected,
                "input_ids": prompt_encodings["input_ids"],
                "attention_mask": prompt_encodings["attention_mask"],
                "chosen_input_ids": chosen_encodings["input_ids"],
                "chosen_attention_mask": chosen_encodings["attention_mask"],
                "rejected_input_ids": rejected_encodings["input_ids"],
                "rejected_attention_mask": rejected_encodings["attention_mask"]
            }
        
        # 創建數據集
        logger.info(f"🔄 Creating dataset from {len(preference_pairs)} preference pairs...")
        dataset = Dataset.from_list(preference_pairs)
        
        # 改進的處理
        dataset = dataset.map(improved_tokenize_function, batched=True)
        
        # 分割數據集
        split_ratio = config['data']['train_test_split']
        if split_ratio > 0 and len(dataset) > 10:  # 只有足夠數據才分割
            dataset = dataset.train_test_split(test_size=split_ratio)
            train_dataset = dataset['train']
            eval_dataset = dataset['test']
            logger.info(f"📊 Train samples: {len(train_dataset)}")
            logger.info(f"📊 Eval samples: {len(eval_dataset)}")
        else:
            train_dataset = dataset
            eval_dataset = None
            logger.info(f"📊 Train samples: {len(train_dataset)}")
        
        logger.info("="*40)
        logger.info("🎯 STEP 4: Setting up GRPO trainer...")
        
        # 步驟4: 設置GRPO訓練器
        os.makedirs(config['training']['output_dir'], exist_ok=True)
        
        # 確保學習率是浮點數
        learning_rate = config['training']['learning_rate']
        if isinstance(learning_rate, str):
            learning_rate = float(learning_rate)
            logger.info(f"🔧 Converted learning_rate from str to float: {learning_rate}")
        
        # GRPO配置 - 【關鍵修復】設置 num_workers 為 0
        grpo_config = GRPOConfig(
            output_dir=str(config['training']['output_dir']),
            per_device_train_batch_size=int(config['training']['batch_size']),
            per_device_eval_batch_size=int(config['training']['batch_size']),
            gradient_accumulation_steps=int(config['training']['gradient_accumulation_steps']),
            num_train_epochs=int(config['training']['num_epochs']),
            learning_rate=float(learning_rate),  # 確保是浮點數
            warmup_steps=int(config['training']['warmup_steps']),
            logging_steps=int(config['training']['logging_steps']),
            save_steps=int(config['training']['save_steps']),
            eval_steps=int(config['training']['eval_steps']),
            bf16=bool(config['system']['bf16']),
            gradient_checkpointing=bool(config['system']['gradient_checkpointing']),
            dataloader_num_workers=0,  # 【關鍵修復】強制設為0避免pickle問題
            remove_unused_columns=False,
            report_to="wandb" if config['system']['use_wandb'] else None
        )
        
        logger.info("✅ GRPO configuration created successfully")
        logger.info(f"🔍 Learning rate type: {type(grpo_config.learning_rate)}, value: {grpo_config.learning_rate}")
        logger.info(f"🔧 DataLoader workers: {grpo_config.dataloader_num_workers} (fixed to avoid pickle error)")
        
        # 創建訓練器 - 使用全局reward function
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            reward_funcs=[global_reward_function],  # 使用全局函數
        )
        logger.info("✅ GRPO trainer created successfully")
        
        logger.info("="*40)
        logger.info("🚀 STEP 5: Starting training...")
        
        # 步驟5: 開始訓練
        logger.info(f"🔥 GPU記憶體: {get_gpu_memory_info()}")
        logger.info("🎯 Starting QUICK TEST GRPO training...")
        
        # 開始訓練
        trainer.train()
        
        logger.info("="*40)
        logger.info("💾 STEP 6: Saving model...")
        
        # 步驟6: 保存模型
        model_save_path = os.path.join(config['training']['output_dir'], "grpo_reasoning_model_50percent")
        trainer.save_model(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        
        logger.info(f"✅ Model saved to: {model_save_path}")
        
        # 生成評估樣本
        logger.info("🧪 Generating evaluation sample...")
        test_sample = preference_pairs[0]
        
        inputs = tokenizer(test_sample["prompt"], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        evaluation_result = {
            "prompt": test_sample["prompt"],
            "chosen": test_sample["chosen"],
            "rejected": test_sample["rejected"],
            "generated": generated_text
        }
        
        with open(f"{config['training']['output_dir']}/evaluation_sample.json", 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        print("="*60)
        print("🎉 QUICK TEST GRPO Training Completed Successfully!")
        print(f"📁 Model saved to: {model_save_path}")
        print(f"📊 Evaluation sample: {config['training']['output_dir']}/evaluation_sample_20percent.json")
        print("🚀 Ready for full training if results look good!")
        print("💡 Key fix: Set dataloader_num_workers=0 to avoid pickle errors")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error("完整錯誤信息:")
        logger.error(traceback.format_exc())
        raise
    
    finally:
        # 清理資源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if 'config' in locals() and config['system']['use_wandb']:
            wandb.finish()
        
        logger.info(f"📁 Main log file: {main_log_file}")
        logger.info(f"📈 Progress log file: {progress_log_file}")
        logger.info(f"❌ Error log file: {error_log_file}")
        logger.info("💾 Saving final log summary...")

if __name__ == "__main__":
    main()
