"""
run_eval.py

Runs ROUGE + BERTScore evaluation inside a SageMaker Training Job container.
Uses the same DLC (pytorch 2.1, transformers 4.36) as the training job.

The test dataset is passed via the 'training' input channel.
The model.tar.gz S3 URI is passed as a hyperparameter.
Results are saved to /opt/ml/model/ (SageMaker uploads this to S3).
"""

import argparse
import json
import logging
import os
import tarfile
import tempfile

import boto3
import evaluate as hf_evaluate
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 16
MAX_GENERATE_LENGTH = 256
BASE_MODEL_NAME = "google/flan-t5-base"


def download_and_extract_model(model_s3_uri: str) -> str:
    """Download model.tar.gz from S3 and extract to a temp directory."""
    tmp_dir = tempfile.mkdtemp()
    tar_path = os.path.join(tmp_dir, "model.tar.gz")
    model_dir = os.path.join(tmp_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Parse s3://bucket/key from the URI
    parts = model_s3_uri.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]

    logger.info(f"Downloading model from s3://{bucket}/{key}")
    boto3.client("s3").download_file(bucket, key, tar_path)

    logger.info(f"Extracting to {model_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(model_dir)

    return model_dir


def load_model(model_dir: str, device: str):
    """Load the merged fine-tuned model."""
    logger.info(f"Loading model from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def load_baseline(device: str):
    """Load unmodified flan-t5-base for comparison."""
    logger.info("Loading baseline model: google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tokenizer


def batch_generate(model, tokenizer, input_ids_list, device: str) -> list:
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


def compute_rouge(predictions, references):
    rouge = hf_evaluate.load("rouge")
    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}


def compute_bertscore(predictions, references):
    bertscore = hf_evaluate.load("bertscore")
    result = bertscore.compute(
        predictions=predictions, references=references, model_type="distilbert-base-uncased"
    )
    return {
        "bertscore_precision": round(sum(result["precision"]) / len(result["precision"]) * 100, 2),
        "bertscore_recall": round(sum(result["recall"]) / len(result["recall"]) * 100, 2),
        "bertscore_f1": round(sum(result["f1"]) / len(result["f1"]) * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_s3_uri", required=True, help="S3 URI to model.tar.gz")
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    parser.add_argument("--num_samples", type=int, default=0, help="0 = use all test samples")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # Load test data
    dataset = load_from_disk(args.data_dir)
    test_ds = dataset["test"]
    if args.num_samples > 0:
        test_ds = test_ds.select(range(min(args.num_samples, len(test_ds))))

    input_ids = test_ds["input_ids"]
    label_ids = test_ds["labels"]

    # Decode references
    ref_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    clean_labels = [
        [t if t != -100 else ref_tokenizer.pad_token_id for t in label]
        for label in label_ids
    ]
    references = ref_tokenizer.batch_decode(clean_labels, skip_special_tokens=True)
    logger.info(f"Test samples: {len(references)}")

    # --- Baseline ---
    logger.info("=== BASELINE EVALUATION ===")
    base_model, base_tok = load_baseline(device)
    base_preds = batch_generate(base_model, base_tok, input_ids, device)
    base_rouge = compute_rouge(base_preds, references)
    logger.info(f"Baseline ROUGE: {base_rouge}")
    del base_model
    torch.cuda.empty_cache()

    # --- Fine-tuned ---
    logger.info("=== FINE-TUNED EVALUATION ===")
    model_dir = download_and_extract_model(args.model_s3_uri)
    ft_model, ft_tok = load_model(model_dir, device)
    ft_preds = batch_generate(ft_model, ft_tok, input_ids, device)
    ft_rouge = compute_rouge(ft_preds, references)
    logger.info(f"Fine-tuned ROUGE: {ft_rouge}")

    logger.info("Computing BERTScore for fine-tuned model ...")
    ft_bertscore = compute_bertscore(ft_preds, references)
    logger.info(f"Fine-tuned BERTScore: {ft_bertscore}")

    # --- Results ---
    results = {
        "num_test_samples": len(references),
        "baseline": base_rouge,
        "finetuned": {**ft_rouge, **ft_bertscore},
        "improvement": {k: round(ft_rouge[k] - base_rouge[k], 2) for k in base_rouge},
        "sample_predictions": [
            {"reference": references[i], "baseline": base_preds[i], "finetuned": ft_preds[i]}
            for i in range(min(5, len(references)))
        ],
    }

    # Print table
    print("\n" + "=" * 70)
    print(f"{'METRIC':<20} {'BASELINE':<15} {'FINE-TUNED':<15} {'IMPROVEMENT':<15}")
    print("-" * 70)
    for m in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
        b = base_rouge.get(m, 0)
        f = ft_rouge.get(m, 0)
        d = f - b
        arrow = "+" if d > 0 else ""
        print(f"{m:<20} {b:<15} {f:<15} {arrow}{d:.2f}")
    print("=" * 70 + "\n")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "eval_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
