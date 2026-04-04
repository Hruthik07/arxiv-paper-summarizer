"""
predict.py

Lightweight local inference helper. Use this to quickly test
the fine-tuned model on a single paper or a JSON file of recent papers.

Usage:
    # Summarize a single paper (inline text)
    python inference/predict.py --text "Your paper abstract here..."

    # Summarize papers from fetch_recent_papers.py output
    python inference/predict.py --json_file data/recent_papers/recent_papers_20250101.json
"""

import argparse
import json
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "google/flan-t5-base"
INPUT_PREFIX = "summarize: "
MAX_INPUT_LENGTH = 1024
MAX_GENERATE_LENGTH = 256


def load_model(model_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = PeftModel.from_pretrained(base, model_dir).to(device)
    model.eval()
    return model, tokenizer, device


def summarize(text: str, model, tokenizer, device: str) -> str:
    inputs = tokenizer(
        INPUT_PREFIX + text,
        return_tensors="pt",
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_GENERATE_LENGTH,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main(model_dir: str, text: str | None, json_file: str | None) -> None:
    logger.info(f"Loading model from {model_dir}")
    model, tokenizer, device = load_model(model_dir)

    if text:
        summary = summarize(text, model, tokenizer, device)
        print("\n--- SUMMARY ---")
        print(summary)
        print("---------------\n")

    elif json_file:
        with open(json_file, encoding="utf-8") as f:
            papers = json.load(f)

        print(f"\nSummarizing {len(papers)} papers from {json_file}\n")
        for i, paper in enumerate(papers, 1):
            summary = summarize(paper["abstract"], model, tokenizer, device)
            print(f"[{i}/{len(papers)}] {paper['title']}")
            print(f"  Published : {paper['published'][:10]}")
            print(f"  arXiv ID  : {paper['arxiv_id']}")
            print(f"  Summary   : {summary}")
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local inference with fine-tuned model")
    parser.add_argument(
        "--model_dir", default="./outputs/model", help="Path to fine-tuned model"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Single paper text to summarize")
    group.add_argument(
        "--json_file",
        help="Path to recent_papers JSON from fetch_recent_papers.py",
    )
    args = parser.parse_args()
    main(args.model_dir, args.text, args.json_file)
