"""
download_dataset.py

Downloads the ccdv/arxiv-summarization dataset from HuggingFace,
takes a subset, and uploads to S3 as Parquet files.

Usage:
    python data/download_dataset.py --s3_bucket YOUR_BUCKET_NAME
"""

import argparse
import logging
import os
import tempfile

import boto3
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Subset sizes
SPLIT_SIZES = {
    "train": 20_000,
    "validation": 2_000,
    "test": 2_000,
}

# Keywords in abstract to keep only AI Engineering papers
AI_KEYWORDS = {
    "neural network", "deep learning", "machine learning", "language model",
    "transformer", "fine-tuning", "llm", "nlp", "natural language",
    "reinforcement learning", "generative", "diffusion", "bert", "gpt",
    "attention mechanism", "training", "gradient", "classification",
    "object detection", "image recognition", "computer vision",
}


def is_ai_paper(example: dict) -> bool:
    """Return True if the abstract mentions AI/ML topics."""
    abstract = example.get("abstract", "").lower()
    return any(kw in abstract for kw in AI_KEYWORDS)


def download_and_filter(split: str, max_samples: int) -> pd.DataFrame:
    """Load a split, filter for AI papers, and return a DataFrame."""
    logger.info(f"Loading split='{split}' from ccdv/arxiv-summarization ...")
    ds = load_dataset(
        "ccdv/arxiv-summarization",
        split=split,
        streaming=True,
    )

    records = []
    for example in tqdm(ds, desc=f"Filtering {split}"):
        if is_ai_paper(example):
            records.append(
                {
                    "article": example["article"],
                    "abstract": example["abstract"],
                }
            )
        if len(records) >= max_samples:
            break

    logger.info(f"Collected {len(records)} AI papers for split='{split}'")
    return pd.DataFrame(records)


def upload_to_s3(df: pd.DataFrame, s3_bucket: str, s3_key: str) -> str:
    """Save DataFrame as Parquet and upload to S3. Returns the S3 URI."""
    tmp_dir = tempfile.gettempdir()
    local_path = os.path.join(tmp_dir, os.path.basename(s3_key))
    df.to_parquet(local_path, index=False)
    logger.info(f"Uploading {local_path} → s3://{s3_bucket}/{s3_key}")
    s3 = boto3.client("s3")
    s3.upload_file(local_path, s3_bucket, s3_key)
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    logger.info(f"Uploaded: {s3_uri}")
    return s3_uri


def main(s3_bucket: str, s3_prefix: str = "arxiv-summarizer/data/raw") -> None:
    uris = {}
    for split, max_samples in SPLIT_SIZES.items():
        hf_split = "validation" if split == "validation" else split
        df = download_and_filter(hf_split, max_samples)
        s3_key = f"{s3_prefix}/{split}.parquet"
        uri = upload_to_s3(df, s3_bucket, s3_key)
        uris[split] = uri

    logger.info("All splits uploaded:")
    for split, uri in uris.items():
        logger.info(f"  {split}: {uri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download arXiv dataset and upload to S3")
    parser.add_argument("--s3_bucket", required=True, help="S3 bucket name")
    parser.add_argument(
        "--s3_prefix",
        default="arxiv-summarizer/data/raw",
        help="S3 key prefix for output files",
    )
    args = parser.parse_args()
    main(args.s3_bucket, args.s3_prefix)
