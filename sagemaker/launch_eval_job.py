"""
launch_eval_job.py

Submits a SageMaker job to evaluate the fine-tuned model on the test set.
Runs both baseline (flan-t5-base) and fine-tuned model, compares ROUGE + BERTScore.

Uses the same DLC container as training (pytorch 2.1, transformers 4.36).

Usage:
    python sagemaker/launch_eval_job.py \
        --s3_bucket arxiv-summarizer-hruthik \
        --role_arn arn:aws:iam::442999093029:role/SageMakerExecutionRole \
        --model_s3_uri s3://arxiv-summarizer-hruthik/arxiv-summarizer/model-output/arxiv-summarizer-finetune-20260415-032320/output/model.tar.gz \
        --wait
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

# Same DLC as training
TRANSFORMERS_VERSION = "4.36"
PYTORCH_VERSION = "2.1"
PYTHON_VERSION = "py310"

# Smaller instance is fine for eval (no training, just forward passes)
EVAL_INSTANCE = "ml.g4dn.xlarge"
JOB_NAME_PREFIX = "arxiv-summarizer-eval"


def get_job_name() -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{JOB_NAME_PREFIX}-{timestamp}"


def launch_eval_job(
    s3_bucket: str,
    role_arn: str,
    region: str,
    model_s3_uri: str,
    processed_prefix: str,
) -> str:
    boto3.setup_default_session(region_name=region)
    sm_session = sagemaker.Session()

    job_name = get_job_name()
    processed_s3_uri = f"s3://{s3_bucket}/{processed_prefix}"
    output_s3_uri = f"s3://{s3_bucket}/arxiv-summarizer/eval-output/"

    logger.info(f"Job name     : {job_name}")
    logger.info(f"Model        : {model_s3_uri}")
    logger.info(f"Test data    : {processed_s3_uri}")
    logger.info(f"Output       : {output_s3_uri}")
    logger.info(f"Instance     : {EVAL_INSTANCE}")

    estimator = HuggingFace(
        entry_point="run_eval.py",
        source_dir="training",            # shares requirements.txt with training
        role=role_arn,
        instance_type=EVAL_INSTANCE,
        instance_count=1,
        transformers_version=TRANSFORMERS_VERSION,
        pytorch_version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        output_path=output_s3_uri,
        base_job_name=JOB_NAME_PREFIX,
        max_run=7200,                     # 2 hour cap — eval should take ~30 min
        hyperparameters={
            "model_s3_uri": model_s3_uri,
        },
        environment={
            "TOKENIZERS_PARALLELISM": "false",
        },
        sagemaker_session=sm_session,
    )

    estimator.fit(
        inputs={"training": processed_s3_uri},
        job_name=job_name,
        wait=False,
    )

    logger.info(f"Eval job submitted: {job_name}")
    logger.info(
        f"Monitor at: https://console.aws.amazon.com/sagemaker/home?"
        f"region={region}#/jobs/{job_name}"
    )
    return job_name


def wait_for_job(job_name: str, region: str) -> str:
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
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SageMaker evaluation job")
    parser.add_argument("--s3_bucket", required=True)
    parser.add_argument("--role_arn", required=True)
    parser.add_argument("--model_s3_uri", required=True, help="S3 URI to model.tar.gz")
    parser.add_argument(
        "--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    parser.add_argument(
        "--processed_prefix",
        default="arxiv-summarizer/data/processed",
    )
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    job_name = launch_eval_job(
        s3_bucket=args.s3_bucket,
        role_arn=args.role_arn,
        region=args.region,
        model_s3_uri=args.model_s3_uri,
        processed_prefix=args.processed_prefix,
    )

    if args.wait:
        final_status = wait_for_job(job_name, args.region)
        print(f"\nFinal job status: {final_status}")
    else:
        print(f"\nJob submitted: {job_name}")
        print("Run with --wait to block until completion.")
