"""
setup_monitor.py

Configures SageMaker Model Monitor on the deployed endpoint:
  1. Enables Data Capture (logs all inputs/outputs to S3)
  2. Creates a Data Quality baseline from training data
  3. Schedules an hourly monitoring job

Prerequisites:
  - Endpoint must already be deployed (run deploy_endpoint.py first)

Usage:
    python sagemaker/setup_monitor.py \
        --s3_bucket YOUR_BUCKET \
        --role_arn arn:aws:iam::ACCOUNT:role/SageMakerRole
"""

import argparse
import json
import logging
import os

import boto3
import sagemaker
from sagemaker.model_monitor import (
    CronExpressionGenerator,
    DataCaptureConfig,
    DefaultModelMonitor,
)
from sagemaker.s3 import S3Uploader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ENDPOINT_NAME = "arxiv-summarizer-endpoint"
MONITOR_JOB_NAME = "arxiv-summarizer-monitor"
MONITOR_INSTANCE = "ml.t3.medium"   # Lightweight — monitoring doesn't need GPU


def enable_data_capture(endpoint_name: str, s3_bucket: str, region: str) -> str:
    """
    Update the endpoint to capture 100% of requests/responses to S3.
    Returns the S3 URI where captured data will be stored.
    """
    capture_s3_uri = f"s3://{s3_bucket}/arxiv-summarizer/data-capture/{endpoint_name}"
    logger.info(f"Data capture destination: {capture_s3_uri}")

    sm = boto3.client("sagemaker", region_name=region)

    # Get current endpoint config name
    endpoint_info = sm.describe_endpoint(EndpointName=endpoint_name)
    current_config = endpoint_info["EndpointConfigName"]
    endpoint_config = sm.describe_endpoint_config(EndpointConfigName=current_config)

    new_config_name = f"{current_config}-with-capture"
    logger.info(f"Creating new endpoint config: {new_config_name}")

    sm.create_endpoint_config(
        EndpointConfigName=new_config_name,
        ProductionVariants=endpoint_config["ProductionVariants"],
        DataCaptureConfig={
            "EnableCapture": True,
            "InitialSamplingPercentage": 100,
            "DestinationS3Uri": capture_s3_uri,
            "CaptureOptions": [
                {"CaptureMode": "Input"},
                {"CaptureMode": "Output"},
            ],
            "CaptureContentTypeHeader": {
                "JsonContentTypes": ["application/json"],
            },
        },
    )

    logger.info(f"Updating endpoint {endpoint_name} to new config ...")
    sm.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=new_config_name,
    )

    # Wait for update to complete
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    logger.info("Data capture enabled.")
    return capture_s3_uri


def create_baseline_statistics(
    s3_bucket: str,
    role_arn: str,
    region: str,
    sm_session: sagemaker.Session,
) -> str:
    """
    Run a baseline job on a sample of the training data to
    establish statistics/constraints for data quality monitoring.
    Returns the S3 URI of the baseline results.
    """
    baseline_data_s3 = f"s3://{s3_bucket}/arxiv-summarizer/data/raw/train.parquet"
    baseline_results_s3 = f"s3://{s3_bucket}/arxiv-summarizer/monitor-baseline"

    monitor = DefaultModelMonitor(
        role=role_arn,
        instance_count=1,
        instance_type=MONITOR_INSTANCE,
        volume_size_in_gb=20,
        sagemaker_session=sm_session,
    )

    logger.info("Running baseline job (computes statistics + constraints) ...")
    monitor.suggest_baseline(
        baseline_dataset=baseline_data_s3,
        dataset_format=sagemaker.model_monitor.DatasetFormat.csv(header=True),
        output_s3_uri=baseline_results_s3,
        wait=True,
        logs=True,
    )
    logger.info(f"Baseline results saved to: {baseline_results_s3}")
    return baseline_results_s3


def schedule_monitoring_job(
    endpoint_name: str,
    s3_bucket: str,
    baseline_results_s3: str,
    role_arn: str,
    region: str,
    sm_session: sagemaker.Session,
) -> None:
    """Schedule an hourly monitoring job on the endpoint."""
    capture_s3_uri = f"s3://{s3_bucket}/arxiv-summarizer/data-capture/{endpoint_name}"
    monitor_output_s3 = f"s3://{s3_bucket}/arxiv-summarizer/monitor-reports"

    monitor = DefaultModelMonitor(
        role=role_arn,
        instance_count=1,
        instance_type=MONITOR_INSTANCE,
        sagemaker_session=sm_session,
    )

    logger.info(f"Scheduling hourly monitoring job for endpoint: {endpoint_name}")
    monitor.create_monitoring_schedule(
        monitor_schedule_name=MONITOR_JOB_NAME,
        endpoint_input=endpoint_name,
        output_s3_uri=monitor_output_s3,
        statistics=f"{baseline_results_s3}/statistics.json",
        constraints=f"{baseline_results_s3}/constraints.json",
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )
    logger.info(f"Monitoring schedule created: {MONITOR_JOB_NAME}")
    logger.info(f"Reports will be saved to: {monitor_output_s3}")


def print_cloudwatch_info(endpoint_name: str, region: str) -> None:
    print("\n" + "=" * 60)
    print("CLOUDWATCH METRICS TO WATCH")
    print("=" * 60)
    metrics = [
        ("Invocations", "Number of endpoint calls"),
        ("ModelLatency", "Time taken by the model (microseconds)"),
        ("OverheadLatency", "SageMaker overhead latency"),
        ("Invocation4XXErrors", "Client errors (bad input)"),
        ("Invocation5XXErrors", "Server errors (model failures)"),
    ]
    namespace = f"/aws/sagemaker/Endpoints/{endpoint_name}"
    print(f"Namespace: {namespace}")
    for metric, description in metrics:
        print(f"  {metric:<30} {description}")
    print(
        f"\nDashboard: https://console.aws.amazon.com/cloudwatch/home?"
        f"region={region}#dashboards:"
    )
    print("=" * 60 + "\n")


def main(s3_bucket: str, role_arn: str, region: str) -> None:
    boto3.setup_default_session(region_name=region)
    sm_session = sagemaker.Session()

    # Step 1: Enable data capture
    capture_s3_uri = enable_data_capture(ENDPOINT_NAME, s3_bucket, region)

    # Step 2: Create baseline (optional — skip if training data not in S3 yet)
    try:
        baseline_s3 = create_baseline_statistics(s3_bucket, role_arn, region, sm_session)

        # Step 3: Schedule monitoring
        schedule_monitoring_job(
            ENDPOINT_NAME, s3_bucket, baseline_s3, role_arn, region, sm_session
        )
    except Exception as e:
        logger.warning(f"Baseline/scheduling skipped: {e}")
        logger.info("Data capture is still enabled — monitoring will collect data.")

    print_cloudwatch_info(ENDPOINT_NAME, region)
    logger.info("Monitoring setup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up SageMaker Model Monitor")
    parser.add_argument("--s3_bucket", required=True)
    parser.add_argument("--role_arn", required=True)
    parser.add_argument(
        "--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    args = parser.parse_args()
    main(args.s3_bucket, args.role_arn, args.region)
