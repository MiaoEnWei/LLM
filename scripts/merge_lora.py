import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base="/home/mew/mev/llm/llama2"                       # 改成你的 Llama2 目录
adapter="out_llama2_medmcqa_lora/adapter"
outdir ="out_llama2_medmcqa_lora/merged"

tok = AutoTokenizer.from_pretrained(base)
tok.save_pretrained(outdir)

model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(outdir)
print("Merged ->", outdir)
