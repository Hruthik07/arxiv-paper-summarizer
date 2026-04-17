# arXiv Paper Summarizer

An end-to-end ML pipeline that fine-tunes Google's `flan-t5-base` (250M parameters) using LoRA to summarize arXiv research papers. Trained on ~31K papers, deployed as a real-time SageMaker endpoint, and tested across multiple scientific domains.

The fine-tuned model scores 3.4x higher on ROUGE-1 compared to the base model out of the box.

---

## Why this exists

AI engineers and researchers spend a significant amount of time reading through arXiv papers to stay current. With thousands of new papers published weekly, manually reading every abstract becomes a bottleneck. This project addresses that by providing an automated summarization service that can:

- Help researchers quickly triage which papers are worth a deep read
- Power internal knowledge bases and daily paper digests for ML teams
- Serve as a cost-effective alternative to large API-based models (GPT-4, Claude) for this narrow task — a 250M parameter model on a single T4 GPU handles it at a fraction of the cost
- Integrate into existing workflows as a microservice (Slack bots, RAG pipelines, research dashboards)

---

## What it does

You give it a paper abstract, it returns a concise summary. Works best on AI/ML papers (since that's what it was trained on) but handles physics, math, and biology papers reasonably well too.

**Input:**
> We propose a novel approach to parameter-efficient fine-tuning of large language models using low-rank adaptation (LoRA). Our method significantly reduces the number of trainable parameters while maintaining model quality on downstream NLP tasks. Experiments on GPT-3 show that LoRA matches or exceeds full fine-tuning performance with up to 10,000x fewer trainable parameters.

**Output:**
> A parameter-efficient fine-tuning method using low-rank adaptation that matches full fine-tuning performance with far fewer trainable parameters.

---

## How well does it work?

I evaluated on 835 held-out test papers. Here's how the fine-tuned model compares to the base flan-t5-base:

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| ROUGE-1 | 12.25 | 41.30 | +29.05 |
| ROUGE-2 | 4.61 | 14.39 | +9.78 |
| ROUGE-L | 9.47 | 23.68 | +14.21 |
| BERTScore F1 | - | 80.40 | - |

I also tested the live endpoint with papers from different fields:

- ML/NLP papers: ROUGE-L around 42-48 (strong, as expected)
- Physics/astronomy papers: ROUGE-L around 42-66 (surprisingly good)
- Math/biology papers: ROUGE-L around 28-30 (decent for out-of-domain)
- Edge cases (very short text, LaTeX-heavy): handled without errors

Latency on a single ml.g4dn.xlarge instance was about 1.9 seconds per request. Under heavy load (20 concurrent requests), requests queue up since the GPU processes one at a time - that's expected for a single-instance setup.

---

## How I built it

### The pipeline

```
arXiv papers  -->  Tokenize  -->  Fine-tune with LoRA  -->  Evaluate  -->  Deploy  -->  Monitor
 (31K papers)     (to S3)        (SageMaker, 9 hrs)       (ROUGE/BERT)   (endpoint)  (CloudWatch)
```

### Training setup

I used LoRA instead of full fine-tuning because it only trains ~0.5% of the model's parameters. This kept the training cost around $12.50 on a single A10G GPU.

- **Model**: google/flan-t5-base (encoder-decoder, 250M params)
- **Method**: LoRA with r=16, alpha=32, targeting the query and value projections
- **Data**: ccdv/arxiv-summarization dataset from HuggingFace
- **Hardware**: ml.g5.xlarge (A10G GPU, 24GB VRAM) on SageMaker
- **Training time**: ~8.9 hours, 3 epochs
- **Precision**: BF16 mixed precision

After training, I merged the LoRA weights back into the base model using `merge_and_unload()`. This means the deployed model is a standard transformers checkpoint - no PEFT library needed at inference time.

### SageMaker DLC compatibility notes

The HuggingFace Deep Learning Container ships with pre-installed versions of PyTorch, transformers, and accelerate that are tightly coupled. These should **not** be listed in `requirements.txt` — doing so causes pip to replace the CUDA-enabled builds with incompatible versions.

Other gotchas:
- `peft==0.10.0` is pinned because later versions import `EncoderDecoderCache` (only available in transformers 4.39+, but the DLC has 4.36)
- transformers 4.36 uses `evaluation_strategy` (not `eval_strategy`) and `tokenizer=` (not `processing_class=`)

---

## Project structure

```
├── data/
│   ├── download_dataset.py        # Pull arXiv dataset from HuggingFace, upload to S3
│   ├── fetch_recent_papers.py     # Grab fresh papers from arXiv API for demo
│   └── preprocess.py              # Tokenize and push to S3
├── training/
│   ├── config.py                  # All hyperparameters in one place
│   ├── train.py                   # LoRA fine-tuning script (runs on SageMaker)
│   ├── run_eval.py                # Evaluation script (also runs on SageMaker)
│   └── requirements.txt           # Only non-DLC dependencies
├── evaluation/
│   ├── evaluate.py                # ROUGE + BERTScore metrics
│   ├── compare_baseline.py        # Head-to-head with the base model
│   └── output/eval_results.json   # The actual numbers
├── sagemaker/
│   ├── launch_training_job.py     # Kicks off training on SageMaker
│   ├── launch_eval_job.py         # Kicks off evaluation on SageMaker
│   ├── deploy_endpoint.py         # Deploys the model + sets up autoscaling
│   └── setup_monitor.py           # Enables data capture and model monitoring
├── inference/
│   ├── app.py                     # FastAPI server for local testing
│   └── predict.py                 # Quick CLI tool to summarize a paper
├── testing/
│   └── test_endpoint.py           # Quality, performance, and load tests
├── monitoring/
│   └── cloudwatch_dashboard.json  # Pre-built CloudWatch dashboard config
└── notebooks/                     # Exploration and analysis notebooks
```

---

## Running it yourself

### What you need

- Python 3.10+
- AWS account with SageMaker access
- IAM role with SageMaker and S3 permissions
- An S3 bucket

```bash
pip install -r requirements.txt
aws configure
```

### Step by step

**1. Get the data**
```bash
python data/download_dataset.py --s3_bucket YOUR_BUCKET
python data/preprocess.py --s3_bucket YOUR_BUCKET
```

**2. Train**
```bash
python sagemaker/launch_training_job.py \
    --s3_bucket YOUR_BUCKET \
    --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
    --wait
```
This takes about 9 hours and costs ~$12.50 on ml.g5.xlarge.

**3. Evaluate**
```bash
python sagemaker/launch_eval_job.py \
    --s3_bucket YOUR_BUCKET \
    --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole \
    --model_s3_uri s3://YOUR_BUCKET/.../model.tar.gz \
    --wait
```

**4. Deploy**
```bash
python sagemaker/deploy_endpoint.py \
    --model_s3_uri s3://YOUR_BUCKET/.../model.tar.gz \
    --role_arn arn:aws:iam::ACCOUNT_ID:role/SageMakerRole
```
Deploys on ml.g4dn.xlarge (~$0.75/hr). Includes autoscaling and a smoke test.

**5. Test**
```bash
python testing/test_endpoint.py --suite all
```

**6. Use it**
```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
response = runtime.invoke_endpoint(
    EndpointName="arxiv-summarizer-endpoint",
    ContentType="application/json",
    Body=json.dumps({"inputs": "summarize: Your paper text here..."}),
)
print(json.loads(response["Body"].read())[0]["summary_text"])
```

**7. Clean up** (important - stop the billing)
```bash
aws sagemaker delete-endpoint --endpoint-name arxiv-summarizer-endpoint
```

---

## Monitoring

I set up a CloudWatch dashboard that tracks invocations, latency percentiles (P50/P95/P99), error rates, and CPU/memory usage. You can import the config from `monitoring/cloudwatch_dashboard.json` into the CloudWatch console.

There's also SageMaker Model Monitor for data drift detection - `sagemaker/setup_monitor.py` configures hourly monitoring jobs and data capture to S3.

---

## Future improvements

- Scale to flan-t5-large (780M params) for higher quality summaries
- Add request batching on the endpoint to improve throughput under concurrent load
- Build a Streamlit/Gradio UI for interactive use
- CI/CD pipeline with GitHub Actions for automated retraining on new data
