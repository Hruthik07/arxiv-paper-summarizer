"""
app.py

FastAPI inference server for local testing of the fine-tuned model.
Test this before deploying to SageMaker.

Usage:
    uvicorn inference.app:app --reload --port 8000

Then test:
    curl -X POST http://localhost:8000/summarize \
         -H "Content-Type: application/json" \
         -d '{"text": "Your paper text here..."}'
"""

import logging
import os
import time

import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="arXiv Paper Summarizer",
    description="Fine-tuned flan-t5-base for AI Engineering paper summarization",
    version="1.0.0",
)

# Path to fine-tuned model — set via env var or default
MODEL_DIR = os.environ.get("MODEL_DIR", "./outputs/model")
BASE_MODEL = "google/flan-t5-base"
MAX_INPUT_LENGTH = 1024
MAX_GENERATE_LENGTH = 256
INPUT_PREFIX = "summarize: "

# Global model state
model = None
tokenizer = None
device = None


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Paper text or abstract to summarize")
    max_length: int = Field(256, ge=64, le=512, description="Max tokens in generated summary")


class SummarizeResponse(BaseModel):
    summary: str
    input_length_chars: int
    generation_time_seconds: float


@app.on_event("startup")
def load_model():
    """Load model once at startup."""
    global model, tokenizer, device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model from {MODEL_DIR} on {device} ...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, MODEL_DIR).to(device)
    model.eval()
    logger.info("Model ready.")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "device": device}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    input_text = INPUT_PREFIX + request.text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = round(time.time() - start, 3)

    return SummarizeResponse(
        summary=summary,
        input_length_chars=len(request.text),
        generation_time_seconds=elapsed,
    )


@app.get("/")
def root():
    return {
        "message": "arXiv Paper Summarizer API",
        "endpoints": {
            "POST /summarize": "Summarize a paper",
            "GET /health": "Health check",
        },
    }
