"""
evaluate.py

Loads the fine-tuned model and runs ROUGE + BERTScore evaluation
on the held-out test set.

Usage:
    python evaluation/evaluate.py \
        --model_dir ./outputs/model \
        --data_dir ./data/processed_local \
        --output_file ./evaluation/results.json
"""

import argparse
import json
import logging
import os

import evaluate as hf_evaluate
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 8
MAX_GENERATE_LENGTH = 256


def load_model_and_tokenizer(model_dir: str):
    """Load fine-tuned LoRA model + tokenizer from directory."""
    logger.info(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # train.py merges LoRA into the base model before saving, so model_dir contains
    # a standard transformers model — load it directly without PeftModel wrapper.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Model loaded on device: {device}")
    return model, tokenizer, device


def generate_summaries(
    model, tokenizer, input_ids_list: list[list[int]], device: str
) -> list[str]:
    """Generate summaries in batches."""
    summaries = []
    for i in tqdm(range(0, len(input_ids_list), BATCH_SIZE), desc="Generating"):
        batch = input_ids_list[i : i + BATCH_SIZE]
        max_len = max(len(x) for x in batch)
        padded = [x + [tokenizer.pad_token_id] * (max_len - len(x)) for x in batch]
        input_tensor = torch.tensor(padded, dtype=torch.long).to(device)
        attention_mask = (input_tensor != tokenizer.pad_token_id).long()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=MAX_GENERATE_LENGTH,
                num_beams=4,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summaries.extend(decoded)
    return summaries


def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L."""
    rouge = hf_evaluate.load("rouge")
    result = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )
    return {k: round(v * 100, 2) for k, v in result.items()}


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    """Compute BERTScore (precision, recall, F1) using distilbert."""
    bertscore = hf_evaluate.load("bertscore")
    result = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type="distilbert-base-uncased",
    )
    return {
        "bertscore_precision": round(sum(result["precision"]) / len(result["precision"]) * 100, 2),
        "bertscore_recall": round(sum(result["recall"]) / len(result["recall"]) * 100, 2),
        "bertscore_f1": round(sum(result["f1"]) / len(result["f1"]) * 100, 2),
    }


def run_evaluation(model_dir: str, data_dir: str, output_file: str) -> dict:
    model, tokenizer, device = load_model_and_tokenizer(model_dir)

    logger.info(f"Loading test dataset from {data_dir}")
    dataset = load_from_disk(data_dir)
    test_ds = dataset["test"]

    # input_ids are already tokenized; extract them
    input_ids_list = test_ds["input_ids"]
    label_ids = test_ds["labels"]

    # Decode ground-truth abstracts (replace -100 with pad_token_id)
    clean_labels = [
        [token if token != -100 else tokenizer.pad_token_id for token in label]
        for label in label_ids
    ]
    references = tokenizer.batch_decode(clean_labels, skip_special_tokens=True)

    logger.info(f"Generating summaries for {len(input_ids_list)} test examples ...")
    predictions = generate_summaries(model, tokenizer, input_ids_list, device)

    logger.info("Computing ROUGE scores ...")
    rouge_scores = compute_rouge(predictions, references)

    logger.info("Computing BERTScore ...")
    bert_scores = compute_bertscore(predictions, references)

    results = {
        "model_dir": model_dir,
        "num_test_samples": len(predictions),
        **rouge_scores,
        **bert_scores,
        "sample_predictions": [
            {"prediction": predictions[i], "reference": references[i]}
            for i in range(min(5, len(predictions)))
        ],
    }

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")
    print_summary(results)
    return results


def print_summary(results: dict) -> None:
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for key, val in results.items():
        if key not in ("sample_predictions", "model_dir"):
            print(f"  {key:<25} {val}")
    print("\nSample predictions:")
    for i, sample in enumerate(results.get("sample_predictions", []), 1):
        print(f"\n  [{i}] Reference : {sample['reference'][:120]}...")
        print(f"      Prediction: {sample['prediction'][:120]}...")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned summarization model")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model")
    parser.add_argument("--data_dir", required=True, help="Path to processed test dataset")
    parser.add_argument(
        "--output_file",
        default="evaluation/results.json",
        help="Where to save evaluation results",
    )
    args = parser.parse_args()
    run_evaluation(args.model_dir, args.data_dir, args.output_file)
