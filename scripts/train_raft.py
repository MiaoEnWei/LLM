# scripts/train_raft.py
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
# 1. 你的本地 GPT-2 路径
MODEL_NAME = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/gpt2"

# 2. 刚才生成的“带外挂”的训练集 (请确认文件名是否一致)
TRAIN_FILE = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/data/medmcqa_raft_train.json"

# 3. 输出模型路径
OUTPUT_DIR = "./gpt2-medmcqa-raft"

# RAFT 需要更长的上下文来容纳 Context，设为 768 或 1024
MAX_LENGTH = 768  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_raft(example):
    # RAFT 核心格式: Context -> Question -> Options -> Answer
    ctx = example.get('context', '')
    q = example.get('question', '')
    # 有些 context 可能很长，这里再次截断以防万一
    if len(ctx) > 500: ctx = ctx[:500]
    
    opts = f"A) {example.get('opa', '')}\nB) {example.get('opb', '')}\nC) {example.get('opc', '')}\nD) {example.get('opd', '')}"
    
    # 答案处理
    cop = example.get('cop')
    ans_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', '0': 'A', '1': 'B', '2': 'C', '3': 'D'}
    ans_char = ans_map.get(cop, '')
    if not ans_char: return ""
        
    # 构造 Prompt：明确告诉模型 "Based on Context..."
    text = f"Context:\n{ctx}\nQuestion: {q}\n{opts}\nAnswer: {ans_char}{tokenizer.eos_token}"
    return text

def tokenize_function(examples):
    formatted_texts = []
    for i in range(len(examples['question'])):
        item = {k: examples[k][i] for k in examples}
        formatted_texts.append(format_raft(item))
        
    encodings = tokenizer(formatted_texts, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

print(f"Loading RAFT dataset from {TRAIN_FILE}...")
dataset = load_dataset('json', data_files=TRAIN_FILE, split='train')

print("Tokenizing...")
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# 训练参数
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # 因为加入了 Context，序列变长，Batch Size 调小防止 OOM
    gradient_accumulation_steps=8, # 保持总 Batch Size 足够大
    num_train_epochs=3,            # 训练 3 轮
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

print("Starting RAFT Training...")
trainer.train()

print(f"Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("RAFT Training Done!")