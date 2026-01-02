# scripts/train_raft_masked.py
import json
import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# ================= 配置 =================
MODEL_NAME = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/gpt2"
TRAIN_FILE = "./data/medmcqa_raft_train.json" # 确保这个文件还在
OUTPUT_DIR = "./gpt2-medmcqa-raft-masked"     # 新的输出目录
MAX_LENGTH = 768

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_and_mask(example):
    # 1. 准备文本
    ctx = example.get('context', '')[:500] # 截断
    q = example.get('question', '')
    opts = f"A) {example.get('opa', '')}\nB) {example.get('opb', '')}\nC) {example.get('opc', '')}\nD) {example.get('opd', '')}"
    
    cop = example.get('cop')
    ans_map = {
    1: 'A', 2: 'B', 3: 'C', 4: 'D',
    '1': 'A', '2': 'B', '3': 'C', '4': 'D',
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D',
    }
    cop = example.get('cop')
    ans_char = ans_map.get(cop) or ans_map.get(str(cop).strip())
    if not ans_char:
        return None

    # 2. 构造 Prompt 的两部分
    # Part A: 提示部分 (Context + Question + Options + "Answer:")
    prompt_text = f"Context:\n{ctx}\nQuestion: {q}\n{opts}\nAnswer:"
    
    # Part B: 答案部分 ( " A <eos>")
    # 注意：在 Answer: 后面加个空格，符合 GPT 分词习惯
    answer_text = f" {ans_char}{tokenizer.eos_token}"
    
    full_text = prompt_text + answer_text
    
    # 3. Tokenize
    encodings = tokenizer(full_text, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    
    # 4. 制作 Labels (关键步骤！)
    input_ids = encodings['input_ids']
    labels = list(input_ids) # 复制一份
    
    # 找到 prompt_text 的长度 (大概位置)
    # 这种方法虽然笨但最稳：先 encode prompt 部分，看有多长
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=MAX_LENGTH)
    prompt_len = len(prompt_enc['input_ids'])
    
    # 将 prompt 部分的 labels 设为 -100 (忽略 Loss)
    # 这样模型只会被惩罚预测错"答案"的时候，这就强迫它必须利用 Context 来推导答案
    for i in range(len(labels)):
        if i < prompt_len:
            labels[i] = -100
        else:
            # 保持原样 (即 Answer 部分)
            pass
            
    encodings['labels'] = labels
    return encodings

def tokenize_function(examples):
    results = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(len(examples['question'])):
        item = {k: examples[k][i] for k in examples}
        out = format_and_mask(item)
        if out:
            results["input_ids"].append(out["input_ids"])
            results["attention_mask"].append(out["attention_mask"])
            results["labels"].append(out["labels"])
    return results

print(f"Loading RAFT dataset from {TRAIN_FILE}...")
dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')

print("Tokenizing and MASKING inputs...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to("cuda" if torch.cuda.is_available() else "cpu")

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    save_total_limit=2,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Starting MASKED RAFT Training...")
trainer.train()

print(f"Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")