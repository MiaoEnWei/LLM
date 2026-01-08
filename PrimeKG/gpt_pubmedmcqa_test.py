# Layer 1: Basic Imports & Configuration
import json
import os
import pickle
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer


# [Configuration Area] Please modify variables below according to your actual paths
GPT2_PATH           = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/gpt2"
MEDMCQA_FILE        = "/media/miaoen/ad4277ac-5cfe-47b0-a2cc-f9e50e0da444/LLM/data/medmcqa/dev.json"

# Knowledge Base File Paths (Ensure these files are in the current directory, or use absolute paths)
FAISS_INDEX_PATH    = "pubmed_qa.index"
DOCS_PKL_PATH       = "pubmed_documents.pkl"

# Embedding Model (Must match the model used when building the index)
EMBED_MODEL_NAME    = "all-MiniLM-L6-v2" 

DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

# RAG Parameters
TOP_K_DOCS          = 2     # Retrieve the top 2 most relevant abstracts
MAX_CTX_CHARS       = 2000  # Maximum context characters (Prevent GPT-2 memory overflow/context limit issues)

print(f"Config OK. DEVICE = {DEVICE}")


# Layer 2: Model Layer (GPT-2 Loading & Fixed Generation Function)
def load_gpt2(model_path: str = GPT2_PATH):
    print(f"Loading GPT-2 from {model_path} ...")
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_path).to(DEVICE)
        print(f"GPT-2 loaded.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading GPT-2: {e}")
        return None, None

tokenizer, model = load_gpt2()

def gpt2_generate(prompt: str, max_new_tokens: int = 120, do_sample: bool = True, temperature: float = 0.7, top_p: float = 0.9):
    """
    General generation function, UserWarning issues fixed.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    
    # Input length truncation protection (GPT-2 context is usually 1024)
    if inputs.shape[1] > 900:
        inputs = inputs[:, -900:]
        
    attention_mask = torch.ones_like(inputs)

    # Dynamically build parameters to avoid errors when passing temperature with do_sample=False
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.eos_token_id,
        "attention_mask": attention_mask,
    }

    # Only pass temperature when sampling is enabled
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(inputs, **gen_kwargs)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Layer 3: Knowledge Base Layer (Load FAISS & Documents)
def load_retrieval_system():
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(DOCS_PKL_PATH):
        print(f"Error: Knowledge base files not found. Please check if {FAISS_INDEX_PATH} and {DOCS_PKL_PATH} exist.")
        return None, None, None

    print(f"Loading Embedding Model: {EMBED_MODEL_NAME} ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    
    print(f"Loading FAISS Index ...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    print(f"Loading Documents ...")
    with open(DOCS_PKL_PATH, "rb") as f:
        documents = pickle.load(f)
        
    print(f"Knowledge Base Loaded! Index size: {index.ntotal}, Docs count: {len(documents)}")
    return embed_model, index, documents

# Initialize global variables
embed_model, faiss_index, doc_store = load_retrieval_system()


# Layer 4: Vector Retrieval RAG -- Semantic Search Core Logic
def get_pubmed_context(question_text: str, top_k: int = TOP_K_DOCS) -> str:
    """
    1. Question to Vector
    2. FAISS Search for Similar Document IDs
    3. Extract Text and Concatenate
    """
    if faiss_index is None:
        return ""

    # 1. Encode
    q_emb = embed_model.encode([question_text], convert_to_numpy=True)
    
    # 2. Search
    distances, indices = faiss_index.search(q_emb, top_k)
    
    # 3. Fetch Text
    retrieved_texts = []
    current_chars = 0
    
    for idx_in_store in indices[0]:
        if idx_in_store == -1: continue # Placeholder when FAISS finds nothing
        
        if idx_in_store >= len(doc_store): continue # Prevent index out of bounds

        doc_content = doc_store[idx_in_store]
        
        # Simple cleaning
        clean_content = doc_content.replace("\n", " ").strip()
        if not clean_content: continue

        # Length check
        if current_chars + len(clean_content) > MAX_CTX_CHARS:
            remaining = MAX_CTX_CHARS - current_chars
            retrieved_texts.append(f"Abstract: {clean_content[:remaining]}...")
            break
        
        retrieved_texts.append(f"Abstract: {clean_content}")
        current_chars += len(clean_content)
    
    if not retrieved_texts:
        return ""
    
    return "\n\n".join(retrieved_texts)


# Layer 5: MedMCQA Data Loading
def load_medmcqa_example(idx: int = 0, file_path: str = MEDMCQA_FILE):
    """
    Read the idx-th sample from the MedMCQA dataset
    """
    with open(file_path, "r", encoding="utf-8") as f:
        line_idx = 0
        for line in f:
            line = line.strip()
            if not line: continue
            if line_idx == idx:
                data = json.loads(line)
                
                # Construct question stem
                q = data.get("question") or data.get("Question") or ""
                
                # Construct options
                options_lines = []
                option_map = {"opa": "A", "opb": "B", "opc": "C", "opd": "D"}
                found = False
                for k, lab in option_map.items():
                    if k in data:
                        found = True
                        options_lines.append(f"{lab}) {data[k]}")
                if not found and "options" in data:
                    for i, opt in enumerate(data["options"]):
                        options_lines.append(f"{chr(ord('A')+i)}) {opt}")
                
                question_text = q
                if options_lines:
                    question_text += "\nOptions:\n" + "\n".join(options_lines)
                
                # Get answer
                answer = data.get("cop") or data.get("answer") or data.get("label")
                return {"raw": data, "question_text": question_text, "answer": answer}
            line_idx += 1
    raise IndexError(f"Index {idx} out of range")


# Layer 6: GPT-2 QA Interface (No RAG vs Vector RAG)
def qa_no_rag(question: str) -> str:
    """
    Baseline: No database query, ask GPT-2 directly
    """
    prompt = (
        "You are a medical exam solver.\n"
        "You will be given one multiple-choice question with options A, B, C, and D.\n"
        "Choose the single best option and reply with ONLY one capital letter: A, B, C, or D.\n"
        "Do not output anything else.\n\n"
        f"{question}\n\n"
        "Answer (A, B, C, or D):"
    )
    # do_sample=False indicates greedy search (deterministic results)
    full_text = gpt2_generate(prompt, max_new_tokens=8, do_sample=False)
    
    # Extract the last letter
    tail = full_text.strip()
    for ch in reversed(tail):
        if ch in ["A", "B", "C", "D"]: return ch
    return tail

def qa_with_rag_vector(question: str):
    """
    Vector RAG: Query Database -> Concatenate Context -> Ask GPT-2
    """
    # 1. Get context
    context = get_pubmed_context(question, top_k=TOP_K_DOCS)
    
    if not context:
        return "", qa_no_rag(question)

    # 2. Construct Prompt
    prompt = (
        "You are a medical exam solver.\n"
        "Below are some relevant research abstracts retrieved from PubMed.\n"
        "Use this context to help answer the question.\n"
        "You will be given one multiple-choice question with options A, B, C, and D.\n"
        "Choose the single best option and reply with ONLY one capital letter: A, B, C, or D.\n"
        "Do not output anything else.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer (A, B, C, or D):"
    )
    
    full_text = gpt2_generate(prompt, max_new_tokens=8, do_sample=False)
    
    tail = full_text.strip()
    for ch in reversed(tail):
        if ch in ["A", "B", "C", "D"]:
            return context, ch

    return context, tail


# Layer 7: Comparison Test Function (Single Item)
def compare_rag_medmcqa_vector(idx: int = 0, max_print_chars: int = 600, save_dir: str = "rag_logs_vector"):
    """
    Run a single test and print a detailed report
    """
    try:
        example = load_medmcqa_example(idx)
    except IndexError:
        print(f"Index {idx} out of range.")
        return

    question = example["question_text"]
    gt_raw = example["answer"]

    # Unify answer format to A/B/C/D
    def _to_letter(x):
        if x is None: return None
        s = str(x).strip()
        if s in ["A", "B", "C", "D"]: return s
        if s.isdigit() and 1 <= int(s) <= 4: return chr(ord("A") + int(s) - 1)
        return s

    gt = _to_letter(gt_raw)

    # 1) No RAG
    ans_no = qa_no_rag(question)
    
    # 2) Vector RAG
    retrieved_ctx, ans_rag = qa_with_rag_vector(question)

    correct_no  = (str(ans_no) == str(gt))
    correct_rag = (str(ans_rag) == str(gt))

    # Print report
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"medmcqa_idx_{idx}.txt")
    
    def _short(s): return s if len(s) <= max_print_chars else s[:max_print_chars] + "..."

    print("=" * 60)
    print(f"MedMCQA Sample idx = {idx}")
    print("Question:")
    print(_short(question))
    print(f"\nCorrect Answer (GT): {gt}")
    print("-" * 60)
    print(f"[No RAG] Prediction: {ans_no} | Correct? {correct_no}")
    print("-" * 60)
    print(f"[Vector RAG] Prediction: {ans_rag} | Correct? {correct_rag}")
    print("\n[Retrieved Context (Top 2)]:")
    if retrieved_ctx:
        print(_short(retrieved_ctx))
    else:
        print("(No relevant documents)")
    print("=" * 60)
    
    # Save to file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Q: {question}\nGT: {gt}\n\nCTX:\n{retrieved_ctx}\n\nPred_No: {ans_no}\nPred_RAG: {ans_rag}")

import pandas as pd
import os
from datetime import datetime # Import datetime module


# Layer 8: Batch Evaluation (Auto Timestamp + Directory Management)
def evaluate_medmcqa_acc(start_idx=0, end_idx=100, output_file=None):
    """
    Batch test and save results.
    Arguments:
        output_file: (Optional) Specify filename. If not provided, automatically generated based on current time.
    """
    
    # --- 1. Automatically build filename with timestamp ---
    if output_file is None:
        # Get current time, format: YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"rag_eval_{timestamp}.csv"
    
    # --- 2. Automatically create results folder (Optional, for tidiness) ---
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created: {output_dir}")
    
    # Combine full path: results/rag_eval_2025xxxx.csv
    full_output_path = os.path.join(output_dir, output_file)
    
    print(f"Start batch evaluation: Index {start_idx} -> {end_idx}")
    print(f"Results will be saved to: {full_output_path}")
    
    results = [] 
    total = 0
    correct_no = 0
    correct_rag = 0
    improved = 0
    worsened = 0
    
    for idx in range(start_idx, end_idx):
        try:
            ex = load_medmcqa_example(idx)
        except: 
            continue
        
        q = ex["question_text"]
        raw_ans = ex["answer"]
        
        # Parse Ground Truth (GT)
        gt = None
        if raw_ans and str(raw_ans).strip() in ["A","B","C","D"]: 
            gt = str(raw_ans).strip()
        elif raw_ans and str(raw_ans).isdigit(): 
            gt = chr(ord("A") + int(raw_ans) - 1)
        
        if not gt: continue 

        # === Core Prediction ===
        pred_no = qa_no_rag(q)
        ctx_rag, pred_rag = qa_with_rag_vector(q)
        
        # === Statistics ===
        is_correct_no = (pred_no == gt)
        is_correct_rag = (pred_rag == gt)
        
        if is_correct_no: correct_no += 1
        if is_correct_rag: correct_rag += 1
        total += 1
        
        status = "Same"
        if not is_correct_no and is_correct_rag:
            status = "Improved"
            improved += 1
        elif is_correct_no and not is_correct_rag:
            status = "Worsened"
            worsened += 1
        
        results.append({
            "Index": idx,
            "Question": q,
            "Ground_Truth": gt,
            "Pred_No_RAG": pred_no,
            "Correct_No_RAG": is_correct_no,
            "Pred_Vector_RAG": pred_rag,
            "Correct_Vector_RAG": is_correct_rag,
            "Status": status,
            "Retrieved_Context": ctx_rag
        })

        print(f"[{idx}] GT:{gt} | NoRAG:{pred_no} {'O' if is_correct_no else 'X'} | RAG:{pred_rag} {'O' if is_correct_rag else 'X'} | {status}")

    # === Calculate Final Metrics ===
    acc_no = correct_no / total if total > 0 else 0
    acc_rag = correct_rag / total if total > 0 else 0
    
    print(f"\nFinal Results ({total} questions):")
    print(f"Pure GPT-2 Accuracy : {acc_no:.4f}")
    print(f"RAG (Vector) Accuracy: {acc_rag:.4f}")
    
    # === Save File ===
    if results:
        df = pd.DataFrame(results)
        df["Global_Acc_No_RAG"] = f"{acc_no:.2%}"
        df["Global_Acc_Vector_RAG"] = f"{acc_rag:.2%}"
        
        # 1. Save detailed CSV
        df.to_csv(full_output_path, index=False, encoding="utf-8-sig")
        print(f"Detailed results saved: {full_output_path}")
        
        # 2. Save Summary CSV (Filename automatically adds _summary)
        base, ext = os.path.splitext(output_file) # Separate filename and extension
        summary_filename = f"{base}_summary{ext}"
        full_summary_path = os.path.join(output_dir, summary_filename)
        
        summary_data = [{
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # Record exact time
            "Range": f"{start_idx}-{end_idx}",
            "Total_Questions": total,
            "Acc_No_RAG": acc_no,
            "Acc_Vector_RAG": acc_rag,
            "Improved_Count": improved,
            "Worsened_Count": worsened
        }]
        pd.DataFrame(summary_data).to_csv(full_summary_path, index=False, encoding="utf-8-sig")
        print(f"Summary statistics saved: {full_summary_path}")
        
    else:
        print("No results generated, skipping save.")


if __name__ == "__main__":
    # Method 1: No arguments passed, automatically generate filename with timestamp
    evaluate_medmcqa_acc(0, 4183)
    # Method 2: If you want to specify a name, you can pass it (will also be saved in results folder)
    # evaluate_medmcqa_acc(0, 50, output_file="my_custom_experiment.csv")