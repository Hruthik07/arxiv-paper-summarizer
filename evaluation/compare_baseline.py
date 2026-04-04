"""
compare_baseline.py

Compares the fine-tuned model vs the unmodified flan-t5-base baseline
on a sample of the test set. Prints a side-by-side metrics table.

Usage:
    python evaluation/compare_baseline.py \
        --model_dir ./outputs/model \
        --data_dir ./data/processed_local \
        --num_samples 200
"""

import argparse
import json
import logging

import evaluate as hf_evaluate
import torch
from datasets import load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 8
MAX_GENERATE_LENGTH = 256
BASE_MODEL_NAME = "google/flan-t5-base"


def load_finetuned(model_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, model_dir).to(device)
    model.eval()
    return model, tokenizer


def load_baseline(device: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def batch_generate(model, tokenizer, input_ids_list: list, device: str) -> list[str]:
    summaries = []
    for i in tqdm(range(0, len(input_ids_list), BATCH_SIZE), desc="Generating"):
        batch = input_ids_list[i : i + BATCH_SIZE]
        max_len = max(len(x) for x in batch)
        padded = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in batch]
        input_tensor = torch.tensor(padded).to(device)
        attention_mask = (input_tensor != tokenizer.pad_token_id).long()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=MAX_GENERATE_LENGTH,
                num_beams=4,
                early_stopping=True,
            )
        summaries.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return summaries


def rouge_scores(predictions: list[str], references: list[str]) -> dict:
    rouge = hf_evaluate.load("rouge")
    r = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in r.items()}


def print_comparison_table(baseline: dict, finetuned: dict) -> None:
    metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    col_w = 18
    print("\n" + "=" * 62)
    print(f"{'METRIC':<20} {'BASELINE (flan-t5-base)':<{col_w}} {'FINE-TUNED (LoRA)':<{col_w}}")
    print("-" * 62)
    for m in metrics:
        b_val = baseline.get(m, 0)
        f_val = finetuned.get(m, 0)
        delta = f_val - b_val
        arrow = "▲" if delta > 0 else "▼"
        print(
            f"{m:<20} {b_val:<{col_w}} {f_val:<{col_w - 6}} {arrow} {abs(delta):.2f}"
        )
    print("=" * 62 + "\n")


def main(model_dir: str, data_dir: str, num_samples: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    dataset = load_from_disk(data_dir)
    test_ds = dataset["test"].select(range(min(num_samples, len(dataset["test"]))))

    input_ids = test_ds["input_ids"]
    label_ids = test_ds["labels"]

    # Decode references
    ref_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    clean = [[t if t != -100 else ref_tokenizer.pad_token_id for t in l] for l in label_ids]
    references = ref_tokenizer.batch_decode(clean, skip_special_tokens=True)

    # --- Baseline ---
    logger.info("Running BASELINE model ...")
    base_model, base_tok = load_baseline(device)
    base_preds = batch_generate(base_model, base_tok, input_ids, device)
    base_scores = rouge_scores(base_preds, references)
    del base_model  # free GPU memory

    # --- Fine-tuned ---
    logger.info("Running FINE-TUNED model ...")
    ft_model, ft_tok = load_finetuned(model_dir, device)
    ft_preds = batch_generate(ft_model, ft_tok, input_ids, device)
    ft_scores = rouge_scores(ft_preds, references)

    print_comparison_table(base_scores, ft_scores)

    # Save comparison
    comparison = {
        "num_samples": len(references),
        "baseline": base_scores,
        "finetuned": ft_scores,
        "improvement": {k: round(ft_scores[k] - base_scores[k], 2) for k in base_scores},
    }
    with open("evaluation/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Comparison saved to evaluation/comparison.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare fine-tuned vs baseline")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()
    main(args.model_dir, args.data_dir, args.num_samples)
