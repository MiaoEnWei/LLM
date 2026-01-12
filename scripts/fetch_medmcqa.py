from datasets import load_dataset
import json, os

LETTER = "ABCD"

def cop_to_idx(cop):
    s = str(cop).strip().lower()
    if s in ['a','b','c','d']: return 'abcd'.index(s)
    if s.isdigit():
        k = int(s)
        if 1 <= k <= 4: return k-1
        if 0 <= k <= 3: return k
    raise ValueError(f"Unrecognized cop: {cop}")

def has_label(ex):
    v = ex.get("cop", None)
    if v is None: return False
    s = str(v).strip().lower()
    return s not in ("", "none", "nan", "-1")

def build_prompt(q, a,b,c,d):
    return ("You are a medical exam solver. Choose the single best option and reply with only one letter.\n"
            f"Question: {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer: ")

def main():
    ds_all = load_dataset("openlifescienceai/medmcqa")  # auto-cache
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/instruct", exist_ok=True)   # SFT (with answers)
    os.makedirs("data/infer", exist_ok=True)      # Inference (no answers)

    stats = {}
    for sp in ["train","validation","test"]:
        ds = ds_all[sp]

        # 1) Save raw
        raw_path = f"data/raw/medmcqa_{sp}.jsonl"
        with open(raw_path, "w", encoding="utf-8") as f:
            for ex in ds:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        # 2) Inference prompts (generated in a unified format for offline prediction)
        infer_path = f"data/infer/medmcqa_{sp}_prompts.jsonl"
        with open(infer_path, "w", encoding="utf-8") as pf:
            for i, ex in enumerate(ds):
                prompt = build_prompt(ex["question"], ex["opa"], ex["opb"], ex["opc"], ex["opd"])
                pf.write(json.dumps({"id": i, "prompt": prompt}, ensure_ascii=False) + "\n")

        # 3) Training/supervised data (write only examples with labels)
        inst_path = f"data/instruct/medmcqa_{sp}.jsonl"
        bad = 0
        with open(inst_path, "w", encoding="utf-8") as f:
            for ex in ds:
                if not has_label(ex):
                    bad += 1
                    continue
                idx = cop_to_idx(ex.get("cop"))
                letter = LETTER[idx]
                prompt = build_prompt(ex["question"], ex["opa"], ex["opb"], ex["opc"], ex["opd"])
                # In SFT, it is common to concatenate "input + correct answer" into a single text field
                f.write(json.dumps({"text": prompt + letter}, ensure_ascii=False) + "\n")

        stats[sp] = {
            "count": len(ds),
            "skipped_no_label": bad,
            "raw": raw_path,
            "instruct": inst_path,
            "infer_prompts": infer_path
        }
    print("DONE")
    print(stats)

if __name__ == "__main__":
    main()
