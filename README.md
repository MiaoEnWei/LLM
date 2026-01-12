# LLM Medical QA: GPT-2 & Llama-2 Prompt Tuning

This project evaluates GPT-2 and Llama-2 on medical QA datasets (PubMedQA, MedMCQA) using Parameter-Efficient Fine-Tuning (Prompt Tuning / LoRA).

Important Note (Models Not Included)

Due to repository size limitations, this repo does NOT include any base model weights (e.g., Llama-2 / GPT-2 checkpoints).

Please download the base models from Hugging Face and place them in your local model directory, then update all script arguments to point to your local model paths:

Llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b

GPT-2: https://huggingface.co/openai-community/gpt2

Recommended local layout (example):

./models/llama2/

./models/gpt2/

All commands below assume you have downloaded the models to local disk and use the local paths accordingly.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
2. Prepare models:
    Place base models in ./gpt2 or ./llama2.
    Place datasets in ./data.

Reproduction Commands (GPT-2 Pipeline)
1. Train Prompt Tuning
Train the soft prompts for 1 epoch.

```bash
python scripts/train/train_prompt_tuning.py \
  --model_name_or_path ./gpt2 \
  --train_file data/official_instruct/medmcqa_train.jsonl \
  --output_dir out_gpt2_official_prompt_tuning_e1 \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 0.03 \
  --num_virtual_tokens 10
```
## Experiment 1: PubMedQA (GPT-2)
1. Train (Prompt Tuning)
Hyperparameters: Epochs=5, Batch=8, LR=5e-4

```bash
python scripts/train/train_prompt_tuning.py \
  --model_name_or_path ./gpt2 \
  --train_file data/official_instruct_pubmedqa/pubmedqa_train.jsonl \
  --output_dir out_gpt2_pubmedqa_official_prompt_tuning \
  --epochs 5 \
  --batch_size 8 \
  --grad_accum 1 \
  --lr 5e-4 \
  --seed 2025
```
2. Evaluation (Generation)
Evaluate ROUGE scores and Decision Consistency.

```bash
python scripts/eval_pubmedqa_gen.py \
  --parquet data/pubmedqa/data/pqa_labeled_test.parquet \
  --model ./gpt2 \
  --adapter out_gpt2_pubmedqa_official_prompt_tuning \
  --limit 500 \
  --max_new_tokens 128 \
  --seed 2025 \
  --quiet \
  --with_decision_acc \
  --local_files_only
```
## Experiment 2: PubMedQA (Llama-2)
Note: Llama-2 evaluation requires FP32 inference to avoid numerical instability (e.g., emoji/gibberish output).

1. Train (Prompt Tuning)
(Example command - adjust batch size based on GPU memory)

```bash
python scripts/train/train_prompt_tuning.py \
  --model_name_or_path ./llama2 \
  --train_file data/official_instruct_pubmedqa/pubmedqa_train.jsonl \
  --output_dir out_llama2_official_prompt_tuning \
  --epochs 5 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 3e-2 \
  --num_virtual_tokens 10
```
2. Evaluation (Enhanced v2 Script)
Includes automatic report generation with [✅/❌] tags and Delta Ranking analysis.

```bash
python scripts/eval_pubmedqa_gen_v2.py \
  --parquet data/pubmedqa/data/pqaa_labeled_test.parquet \
  --model ./llama2 \
  --adapter ./out_llama2_official_prompt_tuning \
  --limit 500 \
  --max_new_tokens 128 \
  --seed 2025 \
  --quiet \
  --with_decision_acc \
  --percentile 5
```

## Experiment 3: MedMCQA (GPT-2)
Commands to reproduce the experiments on the MedMCQA dataset.
1. Train (Prompt Tuning)
Hyperparameters: Epochs=1, LR=0.03

```bash
python scripts/train/train_prompt_tuning.py \
  --model_name_or_path ./gpt2 \
  --train_file data/official_instruct/medmcqa_train.jsonl \
  --output_dir out_gpt2_official_prompt_tuning_e1 \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum 8 \
  --lr 0.03 \
  --num_virtual_tokens 10
```
2. Predict Logits

```bash
python scripts/predict_logits_lora.py \
  --base_model ./gpt2 \
  --adapter out_gpt2_official_prompt_tuning \
  --infile data/official_infer/medmcqa_validation_prompts.jsonl \
  --outfile data/official_infer/medmcqa_validation_preds_prompt_tuning.jsonl \
  --batch_size 8 \
  --local_files_only
```
3. Zero-Shot Evaluation
Note: Using float32 inference for numerical stability.

```bash
python scripts/eval_zero_shot_fc.py \
  --model ./gpt2 \
  --adapter out_gpt2_official_prompt_tuning \
  --val data/official_raw/medmcqa_validation.jsonl \
  --max_len 256 \
  --batch 16 \
  --dtype float32 \
  --device_map auto \
  --calib_n 1500 \
  --seed 42
```
## Experiment 4: MedMCQA (Llama-2)
Commands for inference and evaluation on MedMCQA using Llama-2.

1. Predict Logits

```bash
python scripts/predict_logits_lora.py \
  --base_model ./llama2 \
  --adapter ./out_llama2_official_prompt_tuning \
  --infile data/official_infer/medmcqa_validation_prompts.jsonl \
  --outfile data/official_infer/medmcqa_validation_preds_llama2_prompt_tuning.jsonl \
  --batch_size 1 \
  --local_files_only
```
2. Zero-Shot Evaluation
Note: If numerical instability occurs (NaN/Inf), switch --dtype float16 to --dtype float32.

```bash
python scripts/eval_zero_shot_fc.py \
  --model ./llama2 \
  --adapter ./out_llama2_official_prompt_tuning \
  --val data/official_raw/medmcqa_validation.jsonl \
  --max_len 256 \
  --batch 16 \
  --dtype float16 \
  --device_map auto \
  --calib_n 1500 \
  --seed 42
```