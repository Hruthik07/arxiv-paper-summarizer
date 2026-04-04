"""
deploy_endpoint.py

Deploys the fine-tuned model to a SageMaker real-time endpoint
and runs a quick smoke test.

Prerequisites:
  - Training job must have completed and saved model to S3
  - Run: python sagemaker/launch_training_job.py --wait

Usage:
    python sagemaker/deploy_endpoint.py \
        --model_s3_uri s3://YOUR_BUCKET/arxiv-summarizer/model-output/JOB_NAME/output/model.tar.gz \
        --role_arn arn:aws:iam::ACCOUNT:role/SageMakerRole
"""

import argparse
import json
import logging
import os

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFaceModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT_NAME = "arxiv-summarizer-endpoint"
INFERENCE_INSTANCE = "ml.g4dn.xlarge"

# HuggingFace DLC versions (must match training)
TRANSFORMERS_VERSION = "4.36"
PYTORCH_VERSION = "2.1"
PYTHON_VERSION = "py310"

SAMPLE_INPUT = (
    "summarize: We propose a novel approach to parameter-efficient fine-tuning "
    "of large language models using low-rank adaptation (LoRA). Our method "
    "significantly reduces the number of trainable parameters while maintaining "
    "model quality on downstream NLP tasks. Experiments on GPT-3 show that LoRA "
    "matches or exceeds full fine-tuning performance with up to 10,000x fewer "
    "trainable parameters, enabling fine-tuning on consumer-grade hardware."
)


def deploy_model(model_s3_uri: str, role_arn: str, region: str) -> str:
    """
    Create a SageMaker model and deploy it as a real-time endpoint.
    Returns the endpoint name.
    """
    boto3.setup_default_session(region_name=region)
    sm_session = sagemaker.Session()

    logger.info(f"Creating HuggingFaceModel from: {model_s3_uri}")
    hf_model = HuggingFaceModel(
        model_data=model_s3_uri,
        role=role_arn,
        transformers_version=TRANSFORMERS_VERSION,
        pytorch_version=PYTORCH_VERSION,
        py_version=PYTHON_VERSION,
        env={
            "HF_TASK": "summarization",
            "TOKENIZERS_PARALLELISM": "false",
        },
        sagemaker_session=sm_session,
    )

    logger.info(f"Deploying to endpoint: {ENDPOINT_NAME} on {INFERENCE_INSTANCE} ...")
    predictor = hf_model.deploy(
        initial_instance_count=1,
        instance_type=INFERENCE_INSTANCE,
        endpoint_name=ENDPOINT_NAME,
    )

    logger.info(f"Endpoint deployed: {ENDPOINT_NAME}")
    return ENDPOINT_NAME


def configure_autoscaling(endpoint_name: str, region: str) -> None:
    """Set up Application Auto Scaling on the endpoint variant."""
    autoscaling = boto3.client("application-autoscaling", region_name=region)
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

    logger.info("Registering scalable target ...")
    autoscaling.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=1,
        MaxCapacity=2,
    )

    logger.info("Creating scaling policy (target tracking: 70% CPU) ...")
    autoscaling.put_scaling_policy(
        PolicyName=f"{endpoint_name}-scaling-policy",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 70.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
    logger.info("Autoscaling configured.")


def smoke_test(endpoint_name: str, region: str) -> None:
    """Invoke the endpoint with a sample input and print the response."""
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    logger.info("Running smoke test ...")
    payload = json.dumps({"inputs": SAMPLE_INPUT})
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=payload,
    )
    result = json.loads(response["Body"].read().decode())
    print("\n" + "=" * 60)
    print("SMOKE TEST")
    print("=" * 60)
    print(f"Input : {SAMPLE_INPUT[:100]}...")
    print(f"Output: {result}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy fine-tuned model to SageMaker endpoint")
    parser.add_argument(
        "--model_s3_uri",
        required=True,
        help="S3 URI to model.tar.gz (from training job output)",
    )
    parser.add_argument("--role_arn", required=True)
    parser.add_argument(
        "--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    parser.add_argument(
        "--skip_autoscaling", action="store_true", help="Skip autoscaling setup"
    )
    args = parser.parse_args()

    endpoint_name = deploy_model(args.model_s3_uri, args.role_arn, args.region)

    if not args.skip_autoscaling:
        configure_autoscaling(endpoint_name, args.region)

    smoke_test(endpoint_name, args.region)

    print(f"Endpoint ready: {endpoint_name}")
    print(
        f"Console: https://console.aws.amazon.com/sagemaker/home?"
        f"region={args.region}#/endpoints/{endpoint_name}"
    )
