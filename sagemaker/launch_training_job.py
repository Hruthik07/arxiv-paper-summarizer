"""
launch_training_job.py

Submits a SageMaker Training Job to fine-tune flan-t5-base with LoRA
on the preprocessed arXiv summarization dataset.

Prerequisites:
  1. Run data/download_dataset.py and data/preprocess.py first
  2. Set environment variables:
       AWS_DEFAULT_REGION, S3_BUCKET, SAGEMAKER_ROLE_ARN

Usage:
    python sagemaker/launch_training_job.py \
        --s3_bucket YOUR_BUCKET \
        --role_arn arn:aws:iam::ACCOUNT:role/SageMakerRole
"""

import argparse
import logging
import os
import time

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# SageMaker instance
# GPU (fast): "ml.g4dn.xlarge"  — requires quota increase for new accounts
# CPU (slow, free tier): "ml.m5.xlarge"
TRAINING_INSTANCE = "ml.m5.xlarge"

# HuggingFace DLC image versions (check AWS docs for latest)
TRANSFORMERS_VERSION = "4.36"
PYTORCH_VERSION = "2.1"
PYTHON_VERSION = "py310"

JOB_NAME_PREFIX = "arxiv-summarizer-finetune"


def get_job_name() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{JOB_NAME_PREFIX}-{timestamp}"


def launch_training_job(
    s3_bucket: str,
    role_arn: str,
    region: str,
    processed_prefix: str,
) -> str:
    """
    Submit the SageMaker training job.
    Returns the job name.
    """
    boto3.setup_default_session(region_name=region)
    sm_session = sagemaker.Session()

    job_name = get_job_name()
    processed_s3_uri = f"s3://{s3_bucket}/{processed_prefix}"
    output_s3_uri = f"s3://{s3_bucket}/arxiv-summarizer/model-output/"

    logger.info(f"Job name     : {job_name}")
    logger.info(f"Training data: {processed_s3_uri}")
    logger.info(f"Model output : {output_s3_uri}")
    logger.info(f"Instance     : {TRAINING_INSTANCE}")

    estimator = HuggingFace(
        entry_point="train.py",
        source_dir="training",            # uploads the training/ folder to S3
        role=role_arn,
        instance_type=TRAINING_INSTANCE,
        instance_count=1,
        transformers_version=TRANSFORMERS_VERSION,
        pytorch_version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        output_path=output_s3_uri,
        base_job_name=JOB_NAME_PREFIX,
        hyperparameters={
            # Passed as CLI args to train.py
            # data_dir and output_dir are set via SM env vars automatically
        },
        environment={
            "TOKENIZERS_PARALLELISM": "false",
        },
        sagemaker_session=sm_session,
    )

    # Pass the processed dataset as the 'training' input channel
    # SageMaker copies it to /opt/ml/input/data/training inside the container
    estimator.fit(
        inputs={"training": processed_s3_uri},
        job_name=job_name,
        wait=False,   # Non-blocking: monitor progress in SageMaker console
    )

    logger.info(f"Training job submitted: {job_name}")
    logger.info(
        f"Monitor at: https://console.aws.amazon.com/sagemaker/home?"
        f"region={region}#/jobs/{job_name}"
    )
    return job_name


def wait_for_job(job_name: str, region: str) -> str:
    """Poll until the job completes. Returns final status."""
    sm = boto3.client("sagemaker", region_name=region)
    logger.info(f"Waiting for job: {job_name} ...")
    while True:
        response = sm.describe_training_job(TrainingJobName=job_name)
        status = response["TrainingJobStatus"]
        logger.info(f"  Status: {status}")
        if status in ("Completed", "Failed", "Stopped"):
            if status == "Failed":
                reason = response.get("FailureReason", "Unknown")
                logger.error(f"Job failed: {reason}")
            return status
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SageMaker fine-tuning job")
    parser.add_argument("--s3_bucket", required=True)
    parser.add_argument("--role_arn", required=True, help="SageMaker IAM role ARN")
    parser.add_argument(
        "--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    parser.add_argument(
        "--processed_prefix",
        default="arxiv-summarizer/data/processed",
        help="S3 prefix where preprocessed dataset lives",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Block until training job completes",
    )
    args = parser.parse_args()

    job_name = launch_training_job(
        s3_bucket=args.s3_bucket,
        role_arn=args.role_arn,
        region=args.region,
        processed_prefix=args.processed_prefix,
    )

    if args.wait:
        final_status = wait_for_job(job_name, args.region)
        print(f"\nFinal job status: {final_status}")
    else:
        print(f"\nJob submitted: {job_name}")
        print("Run with --wait to block until completion.")
