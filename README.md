# arXiv AI Engineering Paper Summarizer

End-to-end LLM fine-tuning pipeline — from data collection to production deployment on AWS SageMaker.

Fine-tunes `google/flan-t5-base` (250M params) with LoRA to generate concise summaries of arXiv research papers. Trained on 31,203 papers, evaluated on 835 test samples, and deployed as a real-time SageMaker endpoint.

---

## Results

### Evaluation (835 test samples)

| Metric | Baseline (flan-t5-base) | Fine-tuned (LoRA) | Improvement |
|--------|------------------------|-------------------|-------------|
| ROUGE-1 | 12.25 | **41.30** | +29.05 |
| ROUGE-2 | 4.61 | **14.39** | +9.78 |
| ROUGE-L | 9.47 | **23.68** | +14.21 |
| BERTScore F1 | — | **80.40** | — |

**3.4x improvement on ROUGE-1** over the unmodified base model.

### Endpoint Testing (live SageMaker endpoint)

| Test Suite | Result |
|------------|--------|
| Quality (8 samples across ML, physics, math, biology, edge cases) | ROUGE-1: 45.98, BERTScore F1: 85.19 |
| Performance (serial requests) | P50: 1.89s, P95: 3.58s, 0 errors |
| Load (1/5/10/20 concurrent users) | Throughput: 0.56 req/sec, 0 errors |

### Example

**Input** (paper abstract):
> We propose a novel approach to parameter-efficient fine-tuning of large language models using low-rank adaptation (LoRA). Our method significantly reduces the number of trainable parameters while maintaining model quality on downstream NLP tasks. Experiments on GPT-3 show that LoRA matches or exceeds full fine-tuning performance with up to 10,000x fewer trainable parameters, enabling fine-tuning on consumer-grade hardware.

**Generated Summary**:
> We propose a novel approach to parameter-efficient fine-tuning of large language models using low-rank adaptation (LoRA). Our method significantly reduces the number of trainable parameters while maintaining model quality on downstream NLP tasks. Experiments on GPT-3 show that LoRA matches or exceeds full fine-Tuning performance with up to 10,000x fewer trainable parameter.

---

## Architecture

```
Data Collection    Preprocessing     Fine-Tuning       Evaluation        Deployment       Monitoring
     |                  |                |                  |                 |                |
HuggingFace        Tokenize +      SageMaker Job      ROUGE + BERT     SageMaker        CloudWatch
arXiv dataset      upload to S3    flan-t5 + LoRA     Baseline vs      Real-time        Dashboard
31K papers         1024 tokens     ml.g5.xlarge       Fine-tuned       Endpoint         Latency/Errors
                                   8.9 hrs            835 samples      ml.g4dn.xlarge   CPU/Memory
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | `google/flan-t5-base` (250M params, encoder-decoder) |
| Fine-tuning | LoRA via `peft` — r=16, alpha=32, targets q/v projections |
| Dataset | `ccdv/arxiv-summarization` (31K train, 859 val, 835 test) |
| Training | SageMaker Training Job on ml.g5.xlarge (A10G GPU, BF16) |
| Evaluation | SageMaker Job on ml.g4dn.xlarge (T4 GPU) |
| Deployment | SageMaker Real-time Endpoint with autoscaling (1-2 instances) |
| Local API | FastAPI + Uvicorn |
| Monitoring | SageMaker Model Monitor + CloudWatch Dashboard |
| Testing | Custom quality, performance, and load testing suite |

---

## Project Structure

```
fine-tuning/
├── data/
│   ├── download_dataset.py        # Download arXiv dataset from HuggingFace → S3
│   ├── fetch_recent_papers.py     # Fetch fresh papers from arXiv API
│   └── preprocess.py              # Tokenize and upload to S3
├── training/
│   ├── config.py                  # Hyperparameters (LoRA, training args)
│   ├── train.py                   # LoRA fine-tuning (SageMaker entry point)
│   ├── run_eval.py                # Evaluation script (SageMaker entry point)
│   └── requirements.txt           # DLC-compatible dependencies
├── evaluation/
│   ├── evaluate.py                # ROUGE + BERTScore evaluation
│   ├── compare_baseline.py        # Side-by-side baseline comparison
│   └── output/
│       └── eval_results.json      # Final evaluation metrics
├── sagemaker/
│   ├── launch_training_job.py     # Submit training job to SageMaker
│   ├── launch_eval_job.py         # Submit evaluation job to SageMaker
│   ├── deploy_endpoint.py         # Deploy model + autoscaling + smoke test
│   └── setup_monitor.py           # Data capture + Model Monitor scheduling
├── inference/
│   ├── app.py                     # FastAPI local inference server
│   └── predict.py                 # CLI inference helper
├── testing/
│   └── test_endpoint.py           # Quality, performance, and load testing
├── monitoring/
│   └── cloudwatch_dashboard.json  # Import into CloudWatch console
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_training_results.ipynb
    └── 03_evaluation_analysis.ipynb
```

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
aws configure  # set up AWS credentials
```

You need an AWS account with:
- IAM role with `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`
- An S3 bucket for data and model artifacts

### Step 1 — Collect and Preprocess Data

```bash
python data/download_dataset.py --s3_bucket YOUR_BUCKET
python data/preprocess.py --s3_bucket YOUR_BUCKET
```

### Step 2 — Fine-Tune on SageMaker

```bash
python sagemaker/launch_training_job.py \
    --s3_bucket YOUR_BUCKET \
    --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
    --wait
```

Training runs on ml.g5.xlarge (A10G GPU) for ~8-9 hours. Cost: ~$12.50.

The training script:
- Loads flan-t5-base and applies LoRA adapters (r=16, alpha=32)
- Trains for 3 epochs with gradient accumulation (effective batch size 32)
- Merges LoRA weights into the base model via `merge_and_unload()`
- Saves a standard transformers checkpoint (no PEFT dependency at inference)

### Step 3 — Evaluate

Run evaluation on GPU via SageMaker:

```bash
python sagemaker/launch_eval_job.py \
    --s3_bucket YOUR_BUCKET \
    --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
    --model_s3_uri s3://YOUR_BUCKET/arxiv-summarizer/model-output/JOB_NAME/output/model.tar.gz \
    --wait
```

Or evaluate locally (requires model downloaded to `./outputs/model`):

```bash
python evaluation/evaluate.py --model_dir ./outputs/model --data_dir ./data/processed_local
python evaluation/compare_baseline.py --model_dir ./outputs/model --data_dir ./data/processed_local
```

### Step 4 — Deploy to SageMaker

```bash
python sagemaker/deploy_endpoint.py \
    --model_s3_uri s3://YOUR_BUCKET/arxiv-summarizer/model-output/JOB_NAME/output/model.tar.gz \
    --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole
```

This deploys to ml.g4dn.xlarge (~$0.75/hr), configures autoscaling, and runs a smoke test.

### Step 5 — Test the Endpoint

```bash
# Run all test suites (quality + performance + load)
python testing/test_endpoint.py --suite all

# Or run individual suites
python testing/test_endpoint.py --suite quality
python testing/test_endpoint.py --suite performance
python testing/test_endpoint.py --suite load --concurrency-levels 1,5,10,20
```

### Step 6 — Invoke the Endpoint

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
response = runtime.invoke_endpoint(
    EndpointName="arxiv-summarizer-endpoint",
    ContentType="application/json",
    Body=json.dumps({"inputs": "summarize: Your paper text here..."}),
)
result = json.loads(response["Body"].read())
print(result[0]["summary_text"])
```

### Step 7 — Monitor

```bash
python sagemaker/setup_monitor.py \
    --s3_bucket YOUR_BUCKET \
    --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole
```

Import `monitoring/cloudwatch_dashboard.json` into CloudWatch to see invocations, latency (P50/P95/P99), error rates, and CPU/memory utilization.

### Clean Up

```bash
aws sagemaker delete-endpoint --endpoint-name arxiv-summarizer-endpoint
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Base model | google/flan-t5-base (250M params) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q, v (attention projections) |
| Trainable params | ~0.5% of total |
| Epochs | 3 |
| Batch size | 4 (per device) |
| Gradient accumulation | 8 (effective batch size: 32) |
| Learning rate | 3e-4 |
| Precision | BF16 |
| Max input length | 1024 tokens |
| Max target length | 256 tokens |
| Training instance | ml.g5.xlarge (A10G, 24GB VRAM) |
| Training time | ~8.9 hours |
| Training cost | ~$12.50 |

### Key Implementation Notes

- **DLC compatibility**: The SageMaker HuggingFace DLC (PyTorch 2.1, Transformers 4.36) has pre-installed packages. `torch`, `transformers`, and `accelerate` must NOT be listed in `requirements.txt` to avoid overwriting the CUDA-enabled builds.
- **peft pinning**: `peft==0.10.0` is the last version compatible with transformers 4.36 (later versions import `EncoderDecoderCache` which doesn't exist in 4.36).
- **Model saving**: LoRA weights are merged into the base model via `merge_and_unload()`, producing a standard checkpoint that loads without PEFT at inference time.

---

## Key Concepts Demonstrated

| Concept | Where |
|---------|-------|
| LoRA / PEFT fine-tuning | `training/train.py`, `training/config.py` |
| HuggingFace Seq2SeqTrainer | `training/train.py` |
| Data pipeline (HF → S3) | `data/download_dataset.py`, `data/preprocess.py` |
| SageMaker Training Jobs | `sagemaker/launch_training_job.py` |
| ROUGE + BERTScore evaluation | `evaluation/evaluate.py`, `training/run_eval.py` |
| SageMaker endpoint deployment | `sagemaker/deploy_endpoint.py` |
| Autoscaling configuration | `sagemaker/deploy_endpoint.py` |
| Model Monitor + data capture | `sagemaker/setup_monitor.py` |
| CloudWatch observability | `monitoring/cloudwatch_dashboard.json` |
| FastAPI model serving | `inference/app.py` |
| Endpoint load testing | `testing/test_endpoint.py` |
