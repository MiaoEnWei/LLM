import json
import argparse
import os


def join_contexts(ctx_list, max_chars=4000):
    ctx_list = ctx_list or []
    text = " ".join(ctx_list)
    return text[:max_chars]


def normalize_decision(raw):
    """
    Normalize various formats into three lowercase classes: yes/no/maybe:
    - Starts with 'y' -> yes
    - Starts with 'n' -> no
    - Otherwise -> maybe
    """
    s = (str(raw) or "").strip().lower()
    if s.startswith("y"):
        return "yes"
    if s.startswith("n"):
        return "no"
    return "maybe"


def make_pair(pmid, item):
    q = item.get("QUESTION", "")
    ctxs = item.get("CONTEXTS", []) or []
    long_ans = item.get("LONG_ANSWER", "")

    # Prefer final_decision; if missing, fall back to reasoning_required_pred
    raw_dec = item.get("final_decision") or item.get("reasoning_required_pred") or ""
    dec = normalize_decision(raw_dec)  # yes/no/maybe

    ctx = join_contexts(ctxs)

    # Input: keep consistent with the prompt in eval_pubmedqa_gen.py
    inp = (
        "You are a biomedical QA assistant.\n"
        f"Question: {q}\n"
        f"Context: {ctx}\n\n"
        "Provide a concise rationale (1-2 sentences) and end with a final decision word "
        "among [Yes, No, Maybe].\n"
        "Answer:"
    )

    # Output: long answer + a structured final-decision sentence
    out = f"{long_ans.strip()} Therefore, the final decision is {dec.capitalize()}."

    return {
        "id": pmid,
        "input": inp,
        "output": out,
        "final_decision": dec,
    }


def convert_split(in_json, out_jsonl):
    data = json.load(open(in_json, "r", encoding="utf-8"))
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for pmid, item in data.items():
            rec = make_pair(pmid, item)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved -> {out_jsonl}, rows={len(data)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--dev_json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    convert_split(args.train_json, os.path.join(args.out_dir, "pubmedqa_train.jsonl"))
    convert_split(args.dev_json, os.path.join(args.out_dir, "pubmedqa_dev.jsonl"))


if __name__ == "__main__":
    main()