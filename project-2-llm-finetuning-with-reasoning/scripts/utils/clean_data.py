#!/usr/bin/env python
import pandas as pd
import os
import re

def clean_training_data():
    input_file = "C:/Users/NTHUILST/Ray/DL/data/training_data_improve.csv"
    output_file = "C:/Users/NTHUILST/Ray/DL/data/training_data_cleaned.csv"
    
    print("🧹 訓練數據自動清理工具")
    
    if not os.path.exists(input_file):
        print(f"❌ 找不到文件: {input_file}")
        return False
    
    df = pd.read_csv(input_file)
    original = len(df)
    print(f"📥 原始數據: {original} 行")
    
    # 檢測欄位
    cols = {}
    possible = {
        'question': ['題目', 'question'],
        'option_A': ['選項A', 'option_A'],
        'option_B': ['選項B', 'option_B'],  
        'option_C': ['選項C', 'option_C'],
        'option_D': ['選項D', 'option_D'],
        'answer': ['正確答案', 'answer'],
        'reasoning': ['推理正確答案', 'reasoning']
    }
    
    for field, names in possible.items():
        for name in names:
            if name in df.columns:
                cols[field] = name
                break
    
    if len(cols) < 6:
        print("❌ 缺少必需欄位")
        return False
    
    print("🔧 開始清理...")
    
    # 1. 移除空白行
    df = df.dropna(how='all')
    
    # 2. 清理必需欄位空值
    required = ['question', 'option_A', 'option_B', 'option_C', 'option_D', 'answer']
    for field in required:
        if field in cols:
            col = cols[field]
            before = len(df)
            df = df.dropna(subset=[col])
            removed = before - len(df)
            if removed > 0:
                print(f"   {field}: 移除 {removed} 個空值")
    
    # 3. 修復答案格式
    if 'answer' in cols:
        answer_col = cols['answer']
        def fix_answer(ans):
            if pd.isna(ans):
                return None
            ans_str = str(ans).strip().upper()
            ans_clean = re.sub(r'[^ABCD]', '', ans_str)
            if len(ans_clean) == 1 and ans_clean in ['A','B','C','D']:
                return ans_clean
            for letter in ['A','B','C','D']:
                if letter in ans_str:
                    return letter
            return None
        
        before_fix = len(df)
        df[answer_col] = df[answer_col].apply(fix_answer)
        df = df.dropna(subset=[answer_col])
        fixed = before_fix - len(df)
        if fixed > 0:
            print(f"   答案格式: 修復/移除 {fixed} 個")
    
    # 4. 清理文本內容
    text_fields = ['question', 'option_A', 'option_B', 'option_C', 'option_D']
    for field in text_fields:
        if field in cols:
            col = cols[field]
            before = len(df)
            df = df[df[col].str.len() >= 3]  # 移除過短內容
            removed = before - len(df)
            if removed > 0:
                print(f"   {field}: 移除 {removed} 個過短內容")
            df[col] = df[col].str.strip()  # 清理空格
    
    # 5. 移除重複
    before_dup = len(df)
    df = df.drop_duplicates()
    dup_removed = before_dup - len(df)
    if dup_removed > 0:
        print(f"   移除 {dup_removed} 個重複行")
    
    # 6. 重置索引
    df = df.reset_index(drop=True)
    
    final = len(df)
    removed = original - final
    retention = (final / original) * 100
    
    print(f"\n📊 清理結果:")
    print(f"   原始: {original} 行")
    print(f"   清理後: {final} 行")
    print(f"   移除: {removed} 行")
    print(f"   保留率: {retention:.1f}%")
    
    # 保存清理後的數據
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"💾 已保存清理後數據: {output_file}")
    
    if final < original * 0.5:
        print("⚠️  警告: 超過50%的數據被移除")
        return False
    
    print("✅ 數據清理完成！")
    return True, output_file

if __name__ == "__main__":
    result = clean_training_data()
    if result:
        print("\n🎯 現在可以使用清理後的數據訓練:")
        print("   修改訓練腳本中的文件路徑為: training_data_cleaned.csv")
    else:
        print("\n❌ 數據清理失敗")
