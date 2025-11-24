# scripts/predict_generate_lora.py
# 这是一个使用 model.generate() (生成法) 的评测脚本
# 它比“Logits法”更节省显存，适用于 Llama-2 在 12GB 显卡上的评测

import argparse, json, os, torch, re
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from transformers.utils import logging
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logging.set_verbosity_error()

LETTER = "ABCD"

def extract_letter(txt: str) -> str:
    """从生成的文本中提取第一个 A, B, C, 或 D"""
    txt = txt.strip()
    # 匹配开头的 A, B, C, 或 D (忽略大小写)
    m = re.search(r"^\s*([ABCD])", txt, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "?" # 如果模型生成了无关内容，则记为 '?'

def load_model_and_tokenizer(base_model_path, adapter_path, local_only=True):
    """(V2) 使用 4-bit 量化加载模型"""
    
    if adapter_path and not PEFT_AVAILABLE:
        raise ImportError("检测到 --adapter，但未安装 'peft' 库。请运行: pip install peft")

    print(f"Loading base tokenizer from: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        use_fast=True, 
        local_files_only=local_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # ！！！关键：model.generate() 必须使用 "left" 填充 ！！！
    tokenizer.padding_side = "left" 
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading base model from: {base_model_path} (with 4-bit quantization)")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        local_files_only=local_only
    )
    
    if adapter_path:
        print(f"Loading and applying LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, local_files_only=local_only)
        print("Adapter applied.")
        
    model.eval()
    return model, tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="基座模型路径 (e.g., llama2)")
    ap.add_argument("--adapter", default="", help="LoRA 适配器目录 (e.g., out_llama2_official_lora)")
    ap.add_argument("--infile", required=True, help="data/official_infer/..._prompts.jsonl")
    ap.add_argument("--outfile", required=True, help="输出预测的 .jsonl 文件")
    ap.add_argument("--batch_size", type=int, default=4, help="生成时的批大小 (可以适当调大, e.g., 4 or 8)")
    ap.add_argument("--local_files_only", action="store_true", help="强制只使用本地文件")
    args = ap.parse_args()
    
    args.local_files_only = True # 强制使用本地

    if not args.adapter:
        print("WARN: 未提供 --adapter，将直接使用 --base_model (基座) 进行评测。")

    model, tok = load_model_and_tokenizer(
        args.base_model, 
        args.adapter, 
        args.local_files_only
    )
    device = model.device

    # 加载 prompts
    prompts = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            prompts.append((o["id"], o["prompt"])) # (id, prompt_text)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    w = open(args.outfile, "w", encoding="utf-8")
    
    total_processed = 0
    # 这是一个标准的“生成”循环
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i+args.batch_size]
        batch_ids = [x[0] for x in batch]
        batch_texts = [x[1] for x in batch]

        # 标记化
        inputs = tok(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512 # 截断上下文
        ).to(device)

        # 运行模型生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,  # 我们只需要生成一个词 (A/B/C/D)
                do_sample=False,   # 使用贪心解码 (最快)
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )
        
        # 解码 *新* 生成的 token
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        decoded_texts = tok.batch_decode(new_tokens, skip_special_tokens=True)

        # 写入结果
        for ex_id, gen_text in zip(batch_ids, decoded_texts):
            letter = extract_letter(gen_text) # 提取字母
            rec = {"id": ex_id, "pred_letter": letter, "gen_text": gen_text.strip()}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

        total_processed += len(batch)
        print(f"Processed {total_processed} / {len(prompts)}")

    w.close()
    print(f"Saved: {args.outfile}  total={len(prompts)}")

if __name__ == "__main__":
    main()