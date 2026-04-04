"""
config.py

Central configuration for fine-tuning flan-t5-base with LoRA.
Edit values here — train.py reads from this file.
"""

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    model_name: str = "google/flan-t5-base"
    tokenizer_name: str = "google/flan-t5-base"


# ---------------------------------------------------------------------------
# LoRA / PEFT
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    r: int = 16                            # LoRA rank
    lora_alpha: int = 32                   # LoRA scaling factor (alpha/r = 2)
    target_modules: List[str] = field(     # T5 attention projection layers
        default_factory=lambda: ["q", "v"]
    )
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "SEQ_2_SEQ_LM"       # T5 is encoder-decoder


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    output_dir: str = "/opt/ml/model"      # SageMaker writes model here
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    fp16: bool = False                     # Set True only when using GPU (T4/A10G)
    gradient_accumulation_steps: int = 4   # effective batch = 32
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    predict_with_generate: bool = True     # Required for seq2seq evaluation
    generation_max_length: int = 256
    logging_steps: int = 50
    report_to: str = "none"               # Change to "wandb" if you want W&B


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    # S3 paths — set via environment variables or CLI args
    s3_bucket: str = ""
    processed_prefix: str = "arxiv-summarizer/data/processed"
    max_input_length: int = 1024
    max_target_length: int = 256


# ---------------------------------------------------------------------------
# Convenience: single config object
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
