# train_gpt2.py
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

# =================配置=================
MODEL_NAME = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/gpt2" # 你的本地 GPT-2 路径
TRAIN_FILE = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/data/medmcqa/train.json" # 必须用训练集！
OUTPUT_DIR = "./gpt2-medmcqa-finetuned"
MAX_LENGTH = 512  # 上下文长度

# =================1. 加载 Tokenizer=================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# GPT-2 没有 pad token，必须指定
tokenizer.pad_token = tokenizer.eos_token

# =================2. 数据预处理函数=================
def format_medmcqa(example):
    # 将题目格式化为 GPT-2 喜欢的文本流
    # 格式：Question: ... \n Options: ... \n Answer: A
    q = example.get('question', '')
    opts = f"A) {example.get('opa', '')}\nB) {example.get('opb', '')}\nC) {example.get('opc', '')}\nD) {example.get('opd', '')}"
    
    # 获取正确答案的字母 (cop 是索引, 需要转换)
    cop = example.get('cop') 
    # 有些数据 cop 可能是 int (0-3) 或 string
    ans_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', '0': 'A', '1': 'B', '2': 'C', '3': 'D'}
    ans_char = ans_map.get(cop, '')
    
    if not ans_char:
        return "" # 跳过坏数据
        
    text = f"Question: {q}\n{opts}\nAnswer: {ans_char}{tokenizer.eos_token}"
    return text

def tokenize_function(examples):
    # 批量处理
    formatted_texts = []
    # datasets 库加载 json 后是按列存储的 list
    for i in range(len(examples['question'])):
        # 构造单个样本的 dict 传给 format_medmcqa
        item = {k: examples[k][i] for k in examples}
        formatted_texts.append(format_medmcqa(item))
    
    # 编码
    encodings = tokenizer(
        formatted_texts, 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_LENGTH
    )
    # 对于 Causal LM，labels 就是 input_ids
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

# =================3. 加载与处理数据集=================
print("Loading dataset...")
# 加载前 50000 条做快速验证，如果显卡够强，可以加载全部
raw_dataset = load_dataset('json', data_files=TRAIN_FILE, split='train[:50000]') 

print("Tokenizing dataset...")
tokenized_datasets = raw_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=raw_dataset.column_names
)

# =================4. 加载模型=================
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# =================5. 设置训练参数=================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # 显存不够就改小，比如 4 或 2
    gradient_accumulation_steps=4, # 累积梯度，变相增大 batch size
    num_train_epochs=3,            # 训练 3 轮
    learning_rate=5e-5,            # GPT-2 常用微调学习率
    weight_decay=0.01,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,                     # 开启混合精度加速 (如果 GPU 支持)
    dataloader_num_workers=4,
    remove_unused_columns=False
)

# =================6. 开始训练=================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Starting training...")
trainer.train()

# =================7. 保存模型=================
print(f"Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")