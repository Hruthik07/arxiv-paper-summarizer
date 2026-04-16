"""
train.py

Fine-tunes google/flan-t5-base with LoRA on the arXiv summarization dataset.
Designed to run as a SageMaker Training Job (reads data from /opt/ml/input,
writes model to /opt/ml/model) or locally.

Usage (local):
    python training/train.py \
        --data_dir ./data/processed_local \
        --output_dir ./outputs/model

Usage (SageMaker): launched via sagemaker/launch_training_job.py
"""

import argparse
import logging
import os
import sys

import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Add training dir to path so we can import config
sys.path.insert(0, os.path.dirname(__file__))
from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_lora_model(cfg: Config):
    """Load base model and apply LoRA adapters."""
    logger.info(f"Loading base model: {cfg.model.model_name}")

    if cfg.training.bf16:
        dtype = torch.bfloat16
    elif cfg.training.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.model.model_name,
        torch_dtype=dtype,
    )

    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=cfg.lora.target_modules,
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model.enable_input_require_grads()       # Required before get_peft_model with gradient checkpointing
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_dataset(data_dir: str) -> DatasetDict:
    """Load tokenized dataset from disk."""
    logger.info(f"Loading dataset from {data_dir}")
    return load_from_disk(data_dir)


def train(cfg: Config, data_dir: str, output_dir: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    model = build_lora_model(cfg)
    dataset = load_dataset(data_dir)

    logger.info(
        f"Dataset sizes — train: {len(dataset['train'])}, "
        f"val: {len(dataset['validation'])}, test: {len(dataset['test'])}"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # Compute warmup steps from ratio to avoid deprecation warning
    steps_per_epoch = len(dataset["train"]) // (cfg.training.per_device_train_batch_size * cfg.training.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.training.num_train_epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        fp16=cfg.training.fp16 and torch.cuda.is_available(),
        bf16=cfg.training.bf16 and torch.cuda.is_available(),
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        evaluation_strategy=cfg.training.eval_strategy,  # transformers 4.36 uses evaluation_strategy
        save_strategy=cfg.training.save_strategy,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        predict_with_generate=cfg.training.predict_with_generate,
        generation_max_length=cfg.training.generation_max_length,
        logging_steps=cfg.training.logging_steps,
        report_to=cfg.training.report_to,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # CRITICAL: fixes PEFT/LoRA hang
        dataloader_num_workers=0,          # Prevents DataLoader deadlock in SageMaker
        dataloader_pin_memory=False,       # Prevents pin_memory warning in SageMaker
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,      # transformers 4.36 uses tokenizer=, not processing_class=
        data_collator=data_collator,
    )

    logger.info("Starting training ...")
    trainer.train()

    # Merge LoRA adapter weights into the base model and save as a full model.
    # model.save_pretrained() on a PeftModel saves only the adapter (~8MB), which
    # cannot be loaded by the standard HuggingFace inference pipeline. Merging first
    # produces a deployable pytorch_model.bin that any transformers code can load.
    logger.info(f"Merging LoRA adapter into base model and saving to {output_dir}")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune flan-t5-base with LoRA")
    parser.add_argument(
        "--data_dir",
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
        help="Path to processed HuggingFace dataset",
    )
    parser.add_argument(
        "--output_dir",
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
        help="Directory to save the fine-tuned model",
    )
    args = parser.parse_args()

    cfg = Config()
    cfg.training.output_dir = args.output_dir

    train(cfg, args.data_dir, args.output_dir)
