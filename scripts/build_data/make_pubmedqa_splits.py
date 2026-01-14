# scripts/data/make_pubmedqa_splits.py

# Download PubMedQA pqa_labeled from Hugging Face (train only, 1000 examples)
# Split it into train/validation/test, then save as parquet


import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def main():
    out_dir = os.path.join("data", "pubmedqa_hf", "pqa_labeled_splits")
    os.makedirs(out_dir, exist_ok=True)

    print("[make_pubmedqa] Loading PubMedQA pqa_labeled from HF...")
    ds = load_dataset("pubmed_qa", "pqa_labeled")  # only one split: train
    full = ds["train"]
    n = len(full)
    print(f"[make_pubmedqa] Total examples in pqa_labeled/train: {n}")

    # ---- Define 80% train, 10% val, 10% test ----
    idx = np.arange(n)
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    rng.shuffle(idx)

    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    splits = {
        "train": full.select(train_idx),
        "validation": full.select(val_idx),
        "test": full.select(test_idx),
    }

    for name, split_ds in splits.items():
        print(f"[make_pubmedqa] Saving split: {name}, size={len(split_ds)}")
        df = split_ds.to_pandas()
        table = pa.Table.from_pandas(df)
        pq_path = os.path.join(out_dir, f"{name}.parquet")
        pq.write_table(table, pq_path)
        print(f"[make_pubmedqa]   -> {pq_path}")

    print("[make_pubmedqa] Done.")


if __name__ == "__main__":
    main()
