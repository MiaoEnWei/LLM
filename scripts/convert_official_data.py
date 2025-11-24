# scripts/convert_official_data.py (V2 - 修正版，逐行读取)
import json
import os
from collections import Counter

LETTER = "ABCD"

def build_prompt(q, a, b, c, d):
    """构建一个强指令的 prompt."""
    return (
        "You are a medical exam solver. Choose the single best option and reply with only one letter.\n"
        f"Question: {q}\n"
        f"A) {a}\nB) {b}\nC) {c}\nD) {d}\n"
        "Answer (A, B, C, or D): "
    )

def save_split_to_files(dataset_data, sp_name):
    """
    读取数据列表, 保存为 instruct, infer, 和 raw .jsonl 文件.
    同时返回标签分布.
    """
    inst_path = f"data/official_instruct/medmcqa_{sp_name}.jsonl"
    infer_path = f"data/official_infer/medmcqa_{sp_name}_prompts.jsonl"
    raw_path = f"data/official_raw/medmcqa_{sp_name}.jsonl"
    
    os.makedirs(os.path.dirname(inst_path), exist_ok=True)
    os.makedirs(os.path.dirname(infer_path), exist_ok=True)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    
    label_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'Total': 0, 'Invalid': 0}
    
    with open(inst_path, "w", encoding="utf-8") as f_inst, \
         open(infer_path, "w", encoding="utf-8") as f_infer, \
         open(raw_path, "w", encoding="utf-8") as f_raw:
        
        for i, ex in enumerate(dataset_data):
            # 1. 保存 Raw (用于 eval 脚本读取 'cop')
            f_raw.write(json.dumps(ex, ensure_ascii=False) + "\n")
            
            # 2. 保存 Prompt (用于 predict 脚本)
            prompt = build_prompt(ex["question"], ex["opa"], ex["opb"], ex["opc"], ex["opd"])
            f_infer.write(json.dumps({"id": i, "prompt": prompt}, ensure_ascii=False) + "\n")

            # 3. 保存 Instruct (用于训练)
            cop = ex.get("cop")
            if cop is None:
                label_counts['Invalid'] += 1
                continue 
                
            try:
                # 官方数据集使用 1,2,3,4 作为 cop
                idx = int(cop) - 1 
                if 0 <= idx <= 3:
                    letter = LETTER[idx]
                    f_inst.write(json.dumps({"text": prompt + letter}, ensure_ascii=False) + "\n")
                    label_counts[letter] += 1
                    label_counts['Total'] += 1
                else:
                    raise ValueError
            except Exception:
                label_counts['Invalid'] += 1
                
    return label_counts

def main():
    print("Starting conversion of official MedMCQA data (V2: line-by-line)...")
    
    # 1. 定义你下载的官方文件路径
    train_file_path = "data/medmcqa/train.json"
    dev_file_path = "data/medmcqa/dev.json"

    # 2. 加载 TRAIN data (逐行加载)
    print(f"Loading {train_file_path}...")
    try:
        train_data = []
        with open(train_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line))
                
    except FileNotFoundError:
        print(f"ERROR: 找不到 {train_file_path}. 请确认你已将仓库克隆到 'data/medmcqa/'.")
        return
        
    train_stats = save_split_to_files(train_data, "train")
    print(f"New Train Set ({len(train_data)} examples):")
    print(f"  Labels: {train_stats}")

    # 3. 加载 DEV (Validation) data (逐行加载)
    print(f"\nLoading {dev_file_path}...")
    try:
        dev_data = []
        with open(dev_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                dev_data.append(json.loads(line))

    except FileNotFoundError:
        print(f"ERROR: 找不到 {dev_file_path}. 请确认你已将仓库克隆到 'data/medmcqa/'.")
        return

    val_stats = save_split_to_files(dev_data, "validation")
    print(f"\nNew Validation Set ({len(dev_data)} examples):")
    print(f"  Labels: {val_stats}")

    print(f"\nSUCCESS: 转换后的数据已保存到:")
    print("- data/official_instruct/ (用于训练)")
    print("- data/official_infer/ (用于预测)")
    print("- data/official_raw/ (用于评测)")

if __name__ == "__main__":
    main()