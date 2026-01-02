# scripts/prepare_raft_data.py
import json
import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# ================= 配置 =================
# 原始训练集路径
TRAIN_FILE = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/data/medmcqa/train.json"
# 输出的新训练集路径
OUTPUT_FILE = "./data/medmcqa_raft_train.json"

# RAG 模型与数据
EMBED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("1. Loading Retriever System (MedQuad)...")
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE)
embed_model.eval()

# 加载 MedQuad 知识库
kb_dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
kb_texts = []
for item in kb_dataset:
    q = item.get('Question', '').strip()
    a = item.get('Answer', '').strip()
    if len(a) > 20:
        # 截断过长的答案，只留精华，防止训练时 Token 超标
        kb_texts.append(f"Q: {q}\nA: {a[:400]}")

print(f"   Knowledge Base Size: {len(kb_texts)}")

print("2. Building Index...")
def encode(texts):
    embs = []
    batch_size = 128
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding KB"):
        batch = texts[i:i+batch_size]
        inputs = embed_tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = embed_model(**inputs)
            embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    return np.vstack(embs)

doc_embs = encode(kb_texts)
index = faiss.IndexFlatIP(doc_embs.shape[1])
faiss.normalize_L2(doc_embs)
index.add(doc_embs)

print("3. Processing Training Data (Adding Context)...")
# 加载原始训练集
# 注意：为了演示速度，我们只处理前 20,000 条。
# 如果你想跑全量，把 split='train[:20000]' 改为 split='train' (但这会跑很久)
train_data = load_dataset('json', data_files=TRAIN_FILE, split='train[:20000]')

new_data = []

# 批量检索函数
def retrieve_batch(questions, k=1):
    q_embs = encode(questions)
    faiss.normalize_L2(q_embs)
    D, I = index.search(q_embs, k)
    results = []
    for idx_list in I:
        ctx = ""
        for i in idx_list:
            if i >= 0:
                ctx += f"Ref: {kb_texts[i]}\n"
        results.append(ctx)
    return results

# 分批处理训练集
batch_size = 64
questions = train_data['question']
total_items = len(questions)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for i in tqdm(range(0, total_items, batch_size), desc="Augmenting Data"):
        batch_qs = questions[i:i+batch_size]
        # 检索 Context
        contexts = retrieve_batch(batch_qs, k=2) # 每题检索 2 条知识
        
        # 组合并保存
        for j, ctx in enumerate(contexts):
            idx = i + j
            if idx >= total_items: break
            
            original_item = train_data[idx]
            # 构造新的记录，增加了 'context' 字段
            new_record = {
                "question": original_item['question'],
                "opa": original_item['opa'],
                "opb": original_item['opb'],
                "opc": original_item['opc'],
                "opd": original_item['opd'],
                "cop": original_item['cop'],
                "context": ctx # <--- 关键！
            }
            f.write(json.dumps(new_record) + "\n")

print(f"Done! RAFT training data saved to: {OUTPUT_FILE}")