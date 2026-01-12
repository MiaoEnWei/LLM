# scripts/compare_rag_effect.py
import json

# Please replace these with the two result files you actually saved
# File A: results from pure fine-tuning (or RAFT results with RAG turned off)
FILE_NO_RAG = "eval_gen_results_no_rag.jsonl" # e.g., assume you have this file
# File B: RAFT results with RAG turned on
FILE_WITH_RAG = "raft_eval_results.jsonl"

def load_results(path):
    data = {}
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data[item['id']] = item
    return data

def compare():
    print(f"Comparing:\n A: {FILE_NO_RAG}\n B: {FILE_WITH_RAG}\n")
    
    res_a = load_results(FILE_NO_RAG)
    res_b = load_results(FILE_WITH_RAG)
    
    if not res_a or not res_b:
        print("Need both files to compare.")
        return

    total_changed = 0
    rag_helped = 0
    rag_hurt = 0
    
    # Iterate over common IDs
    common_ids = sorted(list(set(res_a.keys()) & set(res_b.keys())))
    
    print(f"{'ID':<6} | {'Gold':<4} | {'NoRAG':<5} -> {'WithRAG':<7} | {'Result'}")
    print("-" * 50)
    
    for idx in common_ids:
        a = res_a[idx]
        b = res_b[idx]
        
        pred_a = a['pred']
        pred_b = b['pred']
        gold = a['gold'] # assume gold is the same
        
        if pred_a != pred_b:
            total_changed += 1
            status = ""
            
            if pred_a != gold and pred_b == gold:
                status = "FIXED"
                rag_helped += 1
            elif pred_a == gold and pred_b != gold:
                status = "BROKE"
                rag_hurt += 1
            else:
                status = "CHANGED (Both Wrong)"
            
            # Print only the first 20 changed examples
            if total_changed <= 20:
                print(f"{idx:<6} | {gold:<4} | {pred_a:<5} -> {pred_b:<7} | {status}")

    print("-" * 50)
    print(f"Total changed predictions: {total_changed}")
    print(f"RAG helped (Fixed): {rag_helped}")
    print(f"RAG hurt (Broke):   {rag_hurt}")
    print(f"Net improvement:    {rag_helped - rag_hurt}")

if __name__ == "__main__":
    import os
    compare()
