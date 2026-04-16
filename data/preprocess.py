"""
preprocess.py

Reads raw Parquet files from S3, tokenizes with flan-t5-base tokenizer,
and uploads the processed (tokenized) dataset back to S3.

Usage:
    python data/preprocess.py \
        --s3_bucket YOUR_BUCKET \
        --raw_prefix arxiv-summarizer/data/raw \
        --processed_prefix arxiv-summarizer/data/processed
"""

import argparse
import logging
import os
import tempfile
from pathlib import Path

import boto3
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_LENGTH = 2048   # tokens — Step 1: longer input captures more paper content
MAX_TARGET_LENGTH = 256   # tokens — abstract length cap
INPUT_PREFIX = "summarize: "  # T5 instruction prefix


def load_from_s3(s3_bucket: str, s3_key: str) -> pd.DataFrame:
    """Download a Parquet file from S3 and return as DataFrame."""
    local_path = os.path.join(tempfile.gettempdir(), Path(s3_key).name)
    logger.info(f"Downloading s3://{s3_bucket}/{s3_key} → {local_path}")
    boto3.client("s3").download_file(s3_bucket, s3_key, local_path)
    return pd.read_parquet(local_path)


def tokenize_batch(batch: dict, tokenizer: AutoTokenizer) -> dict:
    """
    Tokenize a batch of (article, abstract) pairs.

    Input format:  "summarize: {article_text}"
    Target format: "{abstract_text}"
    """
    inputs = [INPUT_PREFIX + article for article in batch["article"]]
    targets = batch["abstract"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True,
    )

    # as_target_tokenizer() was removed in transformers 4.35; use text_target= instead
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
    )

    # Replace padding token id with -100 so it's ignored in loss
    label_ids = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = label_ids
    return model_inputs


def upload_dataset_to_s3(local_dir: str, s3_bucket: str, s3_prefix: str) -> None:
    """Upload all files in a local directory to S3."""
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_prefix}/{relative}".replace("\\", "/")
            logger.info(f"Uploading {local_path} → s3://{s3_bucket}/{s3_key}")
            s3.upload_file(local_path, s3_bucket, s3_key)


def main(s3_bucket: str, raw_prefix: str, processed_prefix: str) -> None:
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    splits = ["train", "validation", "test"]
    dataset_dict = {}

    for split in splits:
        s3_key = f"{raw_prefix}/{split}.parquet"
        df = load_from_s3(s3_bucket, s3_key)
        logger.info(f"Loaded {len(df)} rows for split='{split}'")

        # Convert to HuggingFace Dataset
        ds = Dataset.from_pandas(df[["article", "abstract"]], preserve_index=False)

        # Tokenize
        logger.info(f"Tokenizing split='{split}' ...")
        ds = ds.map(
            lambda batch: tokenize_batch(batch, tokenizer),
            batched=True,
            batch_size=256,
            remove_columns=["article", "abstract"],
            desc=f"Tokenizing {split}",
        )
        dataset_dict[split] = ds

    full_dataset = DatasetDict(dataset_dict)

    # Save locally then upload to S3
    local_output = os.path.join(tempfile.gettempdir(), "processed_dataset")
    logger.info(f"Saving processed dataset to {local_output}")
    full_dataset.save_to_disk(local_output)

    upload_dataset_to_s3(local_output, s3_bucket, processed_prefix)
    logger.info(f"Processed dataset uploaded to s3://{s3_bucket}/{processed_prefix}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize arXiv dataset and upload to S3")
    parser.add_argument("--s3_bucket", required=True)
    parser.add_argument("--raw_prefix", default="arxiv-summarizer/data/raw")
    parser.add_argument("--processed_prefix", default="arxiv-summarizer/data/processed")
    args = parser.parse_args()
    main(args.s3_bucket, args.raw_prefix, args.processed_prefix)
