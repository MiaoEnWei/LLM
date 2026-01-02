# scripts/eval_generation.py
import json
import os
import torch
import numpy as np
import faiss
import argparse  # <--- 新增：命令行参数解析
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from datasets import load_dataset

# ================= 命令行参数解析 =================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./gpt2-medmcqa-raft-masked", help="模型路径")
parser.add_argument("--no_rag", action="store_true", help="如果加上这个参数，就关闭 RAG")
parser.add_argument("--output_file", type=str, default="eval_result.jsonl", help="输出文件名")
args = parser.parse_args()

# ================= 配置 =================
MODEL_PATH = args.model_path
SAVE_FILE = args.output_file
# 逻辑反转：如果命令行加了 --no_rag，则 USE_RAG 为 False
USE_RAG = not args.no_rag 

MEDMCQA_FILE = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/data/medmcqa/dev.json"
EMBED_MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{'='*40}")
print(f"  MODEL: {MODEL_PATH}")
print(f"  RAG STATUS: {'✅ ON' if USE_RAG else '🚫 OFF (Pure Context)'}")
print(f"  OUTPUT: {SAVE_FILE}")
print(f"{'='*40}")

# ================= 1. 加载模型 =================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ================= 2. 初始化 RAG (仅当 USE_RAG=True) =================
rag_index = None
rag_docs = []
embed_tokenizer = None
embed_model = None

if USE_RAG:
    print("Initializing RAG System (MedQuad)...")
    embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
    embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME).to(DEVICE)
    embed_model.eval()

    print("Loading Knowledge Base...")
    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    for item in dataset:
        q = item.get('Question', '').strip()
        a = item.get('Answer', '').strip()
        if len(a) > 20:
            rag_docs.append(f"[{item.get('qtype', 'General')}] {q}\nAnswer: {a}")

    print("Building Vector Index...")
    def encode(texts):
        embs = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            inputs = embed_tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = embed_model(**inputs)
                embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(embs)

    doc_embs = encode(rag_docs)
    rag_index = faiss.IndexFlatIP(doc_embs.shape[1])
    faiss.normalize_L2(doc_embs)
    rag_index.add(doc_embs)
    print(f"RAG Ready. Knowledge Size: {len(rag_docs)}")

def get_rag_context(question):
    if not USE_RAG: 
        return "" # 如果 RAG 关闭，直接返回空字符串
        
    inputs = embed_tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
    with torch.no_grad():
        q_emb = embed_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
    faiss.normalize_L2(q_emb)
    D, I = rag_index.search(q_emb, 3) 
    ctx = ""
    for idx in I[0]:
        if idx >= 0:
            ctx += f"Ref: {rag_docs[idx][:300]}\n"
    return ctx[:600]

# ================= 3. 评估逻辑 =================
def format_prompt(item, context=""):
    q = item['question']
    opts = f"A) {item.get('opa','')}\nB) {item.get('opb','')}\nC) {item.get('opc','')}\nD) {item.get('opd','')}"
    return f"Context:\n{context}\nQuestion: {q}\n{opts}\nAnswer:"

print(f"Starting Evaluation...")

with open(MEDMCQA_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

total = 0
correct = 0
bad_labels = 0
f_out = open(SAVE_FILE, "w", encoding="utf-8")

ans_map = {
    '0': 'A', '1': 'B', '2': 'C', '3': 'D',
    'a': 'A', 'b': 'B', 'c': 'C', 'd': 'D',
    'opa': 'A', 'opb': 'B', 'opc': 'C', 'opd': 'D'
}

for i, line in enumerate(tqdm(lines)):
    item = json.loads(line)
    
    # 答案解析
    cop = item.get('cop')
    cop_str = str(cop).strip().lower()
    ground_truth = ans_map.get(cop_str)
    if not ground_truth and isinstance(cop, int) and 0 <= cop <= 3:
        ground_truth = ['A', 'B', 'C', 'D'][cop]

    if not ground_truth:
        total += 1
        bad_labels += 1
        continue

    # 这里的 ctx 会根据 USE_RAG 自动变为空或实际内容
    ctx = get_rag_context(item['question'])
    prompt = format_prompt(item, context=ctx)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2, pad_token_id=tokenizer.eos_token_id)
        
    new_tokens = outputs[0][inputs.input_ids.shape[1]:]
    pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    pred_char = pred_text[0].upper() if len(pred_text) > 0 else "X"
    
    if pred_char == ground_truth:
        correct += 1
    total += 1
    
    # 保存
    record = {"id": i, "gold": ground_truth, "pred": pred_char, "correct": (pred_char == ground_truth)}
    f_out.write(json.dumps(record) + "\n")
    
    if i > 0 and i % 500 == 0:
        print(f" Step {i} | Acc: {correct/total:.2%}")

f_out.close()

print(f"\n{'='*30}")
print(f"Mode: {'✅ With RAG' if USE_RAG else '🚫 No RAG'}")
print(f"Correct: {correct} / {total}")
print(f"Accuracy: {correct/total:.2%}")
print(f"{'='*30}")