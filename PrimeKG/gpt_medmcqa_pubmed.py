# Layer 1: Basic Imports & Configuration
import json
import os
import pickle
import torch
import faiss
import pandas as pd
import re  # Added regex library
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#PEFT Import
try:
    from peft import PeftModel
except ImportError:
    raise ImportError("Please install peft first: pip install peft")


# [Configuration Area]
BASE_GPT2_PATH      = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/gpt2"
TUNED_GPT2_PATH     = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/out_gpt2_official_prompt_tuning_e1"
MEDMCQA_FILE        = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/data/medmcqa/dev.json"
FAISS_INDEX_PATH    = "pubmed_qa.index"
DOCS_PKL_PATH       = "pubmed_documents.pkl"
EMBED_MODEL_NAME    = "all-MiniLM-L6-v2" 
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

# RAG Parameters
TOP_K_DOCS          = 2     
MAX_CTX_CHARS       = 1500  # Reduced slightly to leave space for the Prompt

print(f"Config OK. DEVICE = {DEVICE}")


# Layer 2: Model Loading (Base + Tuned)
def load_base_gpt2(path):
    print(f"[Base] Loading from {path} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(path).to(DEVICE)
    model.eval()
    return tokenizer, model

def load_tuned_gpt2(base_path, adapter_path):
    print(f"[Tuned] Loading Base from {base_path} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # Load base model
    base_model = GPT2LMHeadModel.from_pretrained(base_path)
    print(f"[Tuned] Loading Adapter from {adapter_path} ...")
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.to(DEVICE)
        model.eval()
        print(f"[Tuned] Adapter combined successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading Adapter: {e}")
        return None, None

#Load both models
tokenizer_base, model_base = load_base_gpt2(BASE_GPT2_PATH)
tokenizer_tuned, model_tuned = load_tuned_gpt2(BASE_GPT2_PATH, TUNED_GPT2_PATH)


# Layer 3: Knowledge Base & RAG
def load_retrieval_system():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCS_PKL_PATH):
        print(f"Error: Knowledge base files not found.")
        return None, None, None
    print(f"Loading RAG System ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(DOCS_PKL_PATH, "rb") as f:
        documents = pickle.load(f)
    return embed_model, index, documents

embed_model, faiss_index, doc_store = load_retrieval_system()

def get_pubmed_context(question_text: str, top_k: int = TOP_K_DOCS) -> str:
    if faiss_index is None: return ""
    q_emb = embed_model.encode([question_text], convert_to_numpy=True)
    distances, indices = faiss_index.search(q_emb, top_k)
    retrieved_texts = []
    current_chars = 0
    for idx_in_store in indices[0]:
        if idx_in_store == -1 or idx_in_store >= len(doc_store): continue
        clean_content = doc_store[idx_in_store].replace("\n", " ").strip()
        if not clean_content: continue
        if current_chars + len(clean_content) > MAX_CTX_CHARS:
            break
        retrieved_texts.append(f"- {clean_content}") # Use list bullet format for clarity
        current_chars += len(clean_content)
    return "\n".join(retrieved_texts) if retrieved_texts else ""


# Layer 4: MedMCQA Data Loading
def load_medmcqa_example(idx: int = 0):
    with open(MEDMCQA_FILE, "r", encoding="utf-8") as f:
        line_idx = 0
        for line in f:
            line = line.strip()
            if not line: continue
            if line_idx == idx:
                data = json.loads(line)
                q = data.get("question") or data.get("Question") or ""
                options_lines = []
                # Explicitly construct "A) OptionText" format to help the model correspond options
                ops = ["opa", "opb", "opc", "opd"]
                for i, k in enumerate(ops):
                    if k in data:
                        options_lines.append(f"{chr(65+i)}) {data[k]}")
                
                # Question part only
                q_text = q
                # Options part only
                o_text = "\n".join(options_lines)
                
                # Concatenate full Questions+Options
                full_q = f"{q}\n{o_text}"
                
                answer = data.get("cop") or data.get("answer") or data.get("label")
                return {"q_only": q_text, "o_only": o_text, "full_text": full_q, "answer": answer}
            line_idx += 1
    return None


# Layer 5: Intelligent Answer Extraction (Key Upgrade)
def extract_answer_letter(gen_text: str) -> str:
    """
    Extract A/B/C/D from messy output.
    Prioritize matching "A)", "Option A", or "A" at the beginning of the sentence.
    """
    s = gen_text.strip()
    # 1. Prioritize finding strong features like "A)" "B)"
    match = re.search(r'([A-D])\)', s)
    if match: return match.group(1)
    
    # 2. Find "Option A"
    match = re.search(r'Option\s+([A-D])', s, re.IGNORECASE)
    if match: return match.group(1)
    
    # 3. Find letter at the very start (e.g., ": A hyper...")
    # Remove impurities like colons, spaces, etc.
    clean_start = re.sub(r'^[:\s\-\.]+', '', s)
    if clean_start and clean_start[0] in ['A', 'B', 'C', 'D']:
        return clean_start[0]
        
    return "X" # Not found


# Layer 6: Core Solver (Prompt Optimized for Tuned Model)
def solve(model, tokenizer, q_data, context=""):
    """
    Based on Format 3 test results, use 'The answer is' as a guide
    """
    # Construct Prompt
    # Format:
    # [Context (Optional)]
    # Question
    # Options
    # The answer is
    
    prompt = ""
    if context:
        prompt += f"Relevant Medical Information:\n{context}\n\n"
    
    prompt += f"Question: {q_data['q_only']}\n"
    prompt += f"Options:\n{q_data['o_only']}\n"
    prompt += "The answer is" # Key guiding phrase
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    if inputs.input_ids.shape[1] > 950: # Truncation protection
        inputs.input_ids = inputs.input_ids[:, -950:]
        inputs.attention_mask = inputs.attention_mask[:, -950:]

    with torch.no_grad():
        # Generate slightly more tokens to prevent only getting ": "
        outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the newly generated part
    new_text = full_text[len(prompt):]
    
    return extract_answer_letter(new_text), new_text


# Layer 7: Batch Comparison
def run_evaluation(start=0, end=100):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results_comparison", exist_ok=True)
    csv_path = f"results_comparison/eval_final_{timestamp}.csv"
    
    print(f"Starting Evaluation: {start} -> {end}")
    
    results = []
    stats = {"Base_Raw":0, "Base_RAG":0, "Tuned_Raw":0, "Tuned_RAG":0, "Total":0}
    
    for idx in range(start, end):
        ex = load_medmcqa_example(idx)
        if not ex: continue
        
        # Parse GT (Ground Truth)
        gt = str(ex["answer"]).strip()
        if gt.isdigit(): gt = chr(ord('A') + int(gt) - 1)
        if gt not in ['A','B','C','D']: continue
        
        stats["Total"] += 1
        
        # 1. Retrieve
        ctx = get_pubmed_context(ex["full_text"])
        
        # 2. Predict
        # Base
        p_base_raw, _ = solve(model_base, tokenizer_base, ex, context="")
        p_base_rag, _ = solve(model_base, tokenizer_base, ex, context=ctx)
        
        # Tuned (Format 3 Logic)
        p_tuned_raw, raw_out_tuned = solve(model_tuned, tokenizer_tuned, ex, context="")
        p_tuned_rag, _             = solve(model_tuned, tokenizer_tuned, ex, context=ctx)
        
        # 3. Statistics
        c_br = (p_base_raw == gt)
        c_b_rag = (p_base_rag == gt)
        c_tr = (p_tuned_raw == gt)
        c_t_rag = (p_tuned_rag == gt)
        
        if c_br: stats["Base_Raw"] += 1
        if c_b_rag: stats["Base_RAG"] += 1
        if c_tr: stats["Tuned_Raw"] += 1
        if c_t_rag: stats["Tuned_RAG"] += 1
        
        print(f"[{idx}] GT:{gt} | Base:{p_base_raw}/{p_base_rag} | Tuned:{p_tuned_raw}/{p_tuned_rag} | TunedRawOut:{raw_out_tuned.strip()[:10]}")
        
        results.append({
            "Idx": idx, "GT": gt,
            "Base_Raw": p_base_raw, "Base_Raw_OK": c_br,
            "Base_RAG": p_base_rag, "Base_RAG_OK": c_b_rag,
            "Tuned_Raw": p_tuned_raw, "Tuned_Raw_OK": c_tr,
            "Tuned_RAG": p_tuned_rag, "Tuned_RAG_OK": c_t_rag,
            "Context": ctx
        })
        
    # Summary
    total = stats["Total"] if stats["Total"] > 0 else 1
    print("\n" + "="*40)
    print(f"Base Raw : {stats['Base_Raw']/total:.2%}")
    print(f"Base RAG : {stats['Base_RAG']/total:.2%}")
    print(f"Tuned Raw: {stats['Tuned_Raw']/total:.2%}")
    print(f"Tuned RAG: {stats['Tuned_RAG']/total:.2%}")
    print("="*40)
    
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    run_evaluation(0, 50)