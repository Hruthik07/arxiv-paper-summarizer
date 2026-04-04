# arXiv AI Engineering Paper Summarizer

End-to-end LLM fine-tuning project — from data collection to production deployment on AWS SageMaker.

**Task**: Fine-tune `google/flan-t5-base` with LoRA to summarize arXiv research papers on AI Engineering topics (MLOps, LLMs, fine-tuning, deployment).

---

## Pipeline Overview

```
Data Collection → Preprocessing → Fine-Tuning (LoRA) → Evaluation → Deploy → Monitor
     ↓                ↓                 ↓                  ↓           ↓         ↓
HuggingFace     Tokenize + S3    SageMaker Job       ROUGE/BERT   SM Endpoint  CloudWatch
arXiv API                        flan-t5 + LoRA      Baseline vs    FastAPI    Model Monitor
                                                      Fine-tuned
```

---

## Example Output

**Input** (paper text):
> We propose a novel approach to parameter-efficient fine-tuning using low-rank adaptation (LoRA). Our method reduces trainable parameters by 10,000x while maintaining model quality on downstream NLP tasks...

**Generated Summary**:
> LoRA reduces trainable parameters significantly while matching full fine-tuning performance, enabling LLM fine-tuning on consumer hardware.

---

## Evaluation Results

| Metric    | Baseline (flan-t5-base) | Fine-tuned (LoRA) | Improvement |
|-----------|------------------------|-------------------|-------------|
| ROUGE-1   | ~28.4                  | ~35.2             | +6.8        |
| ROUGE-2   | ~8.1                   | ~12.3             | +4.2        |
| ROUGE-L   | ~22.6                  | ~29.1             | +6.5        |
| BERTScore | ~82.1                  | ~86.4             | +4.3        |

*Results are illustrative — run evaluation scripts to get your actual numbers.*

---

## Tech Stack

| Component     | Technology                              |
|---------------|-----------------------------------------|
| Base Model    | `google/flan-t5-base` (250M params)     |
| Fine-tuning   | LoRA (`peft`) — only 0.5% params trained|
| Dataset       | `ccdv/arxiv-summarization` (HuggingFace)|
| Training      | SageMaker Training Job (ml.g4dn.xlarge) |
| Deployment    | SageMaker Real-time Endpoint            |
| API           | FastAPI + Uvicorn                       |
| Monitoring    | SageMaker Model Monitor + CloudWatch    |

---

## Project Structure

```
fine-tuning/
├── data/
│   ├── download_dataset.py       # HuggingFace dataset → S3
│   ├── fetch_recent_papers.py    # arXiv API → fresh papers for demo
│   └── preprocess.py             # Tokenize + upload to S3
├── training/
│   ├── config.py                 # All hyperparameters in one place
│   ├── train.py                  # LoRA fine-tuning (SageMaker entry point)
│   └── requirements.txt
├── evaluation/
│   ├── evaluate.py               # ROUGE + BERTScore on test set
│   └── compare_baseline.py       # Side-by-side vs base model
├── sagemaker/
│   ├── launch_training_job.py    # Submit SageMaker training job
│   ├── deploy_endpoint.py        # Deploy + autoscaling
│   └── setup_monitor.py          # Data capture + Model Monitor
├── inference/
│   ├── app.py                    # FastAPI local server
│   └── predict.py                # CLI inference helper
├── monitoring/
│   └── cloudwatch_dashboard.json # Import into CloudWatch
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_results.ipynb
│   └── 03_evaluation_analysis.ipynb
└── requirements.txt
```

---

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
aws configure   # set up AWS credentials
```

### Step 1 — Collect Data
```bash
# Download from HuggingFace and upload to S3
python data/download_dataset.py --s3_bucket YOUR_BUCKET

# Tokenize for training
python data/preprocess.py --s3_bucket YOUR_BUCKET

# (Optional) Fetch fresh arXiv papers for demo
python data/fetch_recent_papers.py --max_results 50
```

### Step 2 — Fine-Tune on SageMaker
```bash
python sagemaker/launch_training_job.py \
    --s3_bucket YOUR_BUCKET \
    --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
    --wait
```
Expected training time: ~2 hours on `ml.g4dn.xlarge` (~$1.50 total).

### Step 3 — Evaluate
```bash
# Download model from S3 first, then:
python evaluation/evaluate.py \
    --model_dir ./outputs/model \
    --data_dir ./data/processed_local

python evaluation/compare_baseline.py \
    --model_dir ./outputs/model \
    --data_dir ./data/processed_local
```

### Step 4 — Test Locally (FastAPI)
```bash
MODEL_DIR=./outputs/model uvicorn inference.app:app --reload

# Test it:
curl -X POST http://localhost:8000/summarize \
     -H "Content-Type: application/json" \
     -d '{"text": "Paste your paper text here..."}'
```

### Step 5 — Deploy to SageMaker
```bash
python sagemaker/deploy_endpoint.py \
    --model_s3_uri s3://YOUR_BUCKET/arxiv-summarizer/model-output/JOB_NAME/output/model.tar.gz \
    --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole
```

### Step 6 — Set Up Monitoring
```bash
python sagemaker/setup_monitor.py \
    --s3_bucket YOUR_BUCKET \
    --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole
```
Then import `monitoring/cloudwatch_dashboard.json` into CloudWatch.

---

## AWS Setup Checklist

- [ ] AWS account with SageMaker access
- [ ] IAM role with `AmazonSageMakerFullAccess` + `AmazonS3FullAccess`
- [ ] S3 bucket created: `s3://YOUR_BUCKET/`
- [ ] `aws configure` completed with your credentials

---

## Key Concepts Demonstrated

| Concept              | Where                                      |
|----------------------|--------------------------------------------|
| LoRA / PEFT          | `training/train.py` + `training/config.py` |
| HuggingFace Trainer  | `training/train.py`                        |
| Data pipeline        | `data/` folder                             |
| SageMaker Training   | `sagemaker/launch_training_job.py`         |
| ROUGE / BERTScore    | `evaluation/evaluate.py`                   |
| Model deployment     | `sagemaker/deploy_endpoint.py`             |
| Autoscaling          | `sagemaker/deploy_endpoint.py`             |
| Model Monitor        | `sagemaker/setup_monitor.py`               |
| CloudWatch metrics   | `monitoring/cloudwatch_dashboard.json`     |
| FastAPI serving      | `inference/app.py`                         |
