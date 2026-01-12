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

#  Configuration 
# 1. Your local GPT-2 path
MODEL_NAME = "/LLM/gpt2"

# 2. The “augmented” training set generated earlier (please confirm the filename matches)
TRAIN_FILE = "/LLM/data/medmcqa_raft_train.json"

# 3. Output model path
OUTPUT_DIR = "./gpt2-medmcqa-raft"

# RAFT needs a longer context window to accommodate Context; set to 768 or 1024
MAX_LENGTH = 768  

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_raft(example):
    # RAFT core format: Context -> Question -> Options -> Answer
    ctx = example.get('context', '')
    q = example.get('question', '')
    # Some contexts may be very long; truncate again here just in case
    if len(ctx) > 500: ctx = ctx[:500]
    
    opts = f"A) {example.get('opa', '')}\nB) {example.get('opb', '')}\nC) {example.get('opc', '')}\nD) {example.get('opd', '')}"
    
    # Answer handling
    cop = example.get('cop')
    ans_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', '0': 'A', '1': 'B', '2': 'C', '3': 'D'}
    ans_char = ans_map.get(cop, '')
    if not ans_char: return ""
        
    # Build the prompt: explicitly tells the model "Based on Context..."
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

# Training hyperparameters
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # With Context added the sequence is longer; reduce batch size to avoid OOM
    gradient_accumulation_steps=8, # Keep the effective batch size large enough
    num_train_epochs=3,            # Train for 3 epochs
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