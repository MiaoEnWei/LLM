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

# ================= Configuration =================
MODEL_NAME = "/LLM/gpt2" # Path to your local GPT-2
TRAIN_FILE = "/LLM/data/medmcqa/train.json" # Must use the training set!
OUTPUT_DIR = "./gpt2-medmcqa-finetuned"
MAX_LENGTH = 512  # Context length

# ================= 1. Load Tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# GPT-2 does not have a pad token, so we must specify one
tokenizer.pad_token = tokenizer.eos_token

# ================= 2. Data Preprocessing Function =================
def format_medmcqa(example):
    # Format the question into a text stream suitable for GPT-2
    # Format: Question: ... \n Options: ... \n Answer: A
    q = example.get('question', '')
    opts = f"A) {example.get('opa', '')}\nB) {example.get('opb', '')}\nC) {example.get('opc', '')}\nD) {example.get('opd', '')}"
    
    # Get the character for the correct answer (cop is an index, needs conversion)
    cop = example.get('cop') 
    # cop in some data might be an int (0-3) or a string
    ans_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', '0': 'A', '1': 'B', '2': 'C', '3': 'D'}
    ans_char = ans_map.get(cop, '')
    
    if not ans_char:
        return "" # Skip corrupted/bad data
        
    text = f"Question: {q}\n{opts}\nAnswer: {ans_char}{tokenizer.eos_token}"
    return text

def tokenize_function(examples):
    # Batch processing
    formatted_texts = []
    # When datasets library loads json, it is a list stored by columns
    for i in range(len(examples['question'])):
        # Construct a dict for a single sample and pass it to format_medmcqa
        item = {k: examples[k][i] for k in examples}
        formatted_texts.append(format_medmcqa(item))
    
    # Encoding
    encodings = tokenizer(
        formatted_texts, 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_LENGTH
    )
    # For Causal LM, labels are a copy of input_ids
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

# ================= 3. Load and Process Dataset =================
print("Loading dataset...")
# Load the first 50,000 records for quick verification. If GPU is strong enough, load all.
raw_dataset = load_dataset('json', data_files=TRAIN_FILE, split='train[:50000]') 

print("Tokenizing dataset...")
tokenized_datasets = raw_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=raw_dataset.column_names
)

# ================= 4. Load Model =================
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# ================= 5. Set Training Arguments =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # Reduce if VRAM is insufficient (e.g., 4 or 2)
    gradient_accumulation_steps=4, # Accumulate gradients to effectively increase batch size
    num_train_epochs=3,             # Train for 3 epochs
    learning_rate=5e-5,            # Common fine-tuning learning rate for GPT-2
    weight_decay=0.01,
    logging_steps=100,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,                     # Enable mixed precision acceleration (if supported by GPU)
    dataloader_num_workers=4,
    remove_unused_columns=False
)

# ================= 6. Start Training =================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("Starting training...")
trainer.train()

# ================= 7. Save Model =================
print(f"Saving model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")