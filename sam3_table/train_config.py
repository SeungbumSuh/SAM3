"""Pydantic models for SAM3 LoRA training configuration."""

from pathlib import Path
from typing import Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field


# --- Model Settings ---


class ModelConfig(BaseModel):
    """Model settings."""

    name: str = "facebook/sam3"
    cache_dir: Optional[str] = None


# --- LoRA Settings ---


class LoRAConfig(BaseModel):
    """LoRA settings for SAM3 architecture."""

    rank: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "qkv",
            "proj",
            "fc1",
            "fc2",
            "c_fc",
            "c_proj",
            "linear1",
            "linear2",
        ]
    )
    apply_to_vision_encoder: bool = True
    apply_to_text_encoder: bool = True
    apply_to_geometry_encoder: bool = True
    apply_to_detr_encoder: bool = True
    apply_to_detr_decoder: bool = True
    apply_to_mask_decoder: bool = True


# --- Training Settings ---


class TrainingConfig(BaseModel):
    """Training settings."""

    data_dir: Union[str, Path] = "/workspace/data"
    batch_size: int = 4
    num_workers: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_epochs: int = 100
    warmup_steps: int = 200
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 5
    mixed_precision: Literal["bf16", "fp16", "no"] = "bf16"
    seed: int = 42
    gradient_accumulation_steps: int = 8


# --- Output Settings ---


class OutputConfig(BaseModel):
    """Output settings."""

    output_dir: Union[str, Path] = "outputs/sam3_lora_full"
    logging_dir: str = "logs"
    save_lora_only: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None


# --- Evaluation Settings ---


class EvaluationConfig(BaseModel):
    """Evaluation settings."""

    metric: str = "iou"
    save_predictions: bool = False
    compute_metrics_during_training: bool = True


# --- Hardware Settings ---


class HardwareConfig(BaseModel):
    """Hardware settings."""

    device: str = "cuda"
    dataloader_pin_memory: bool = True
    use_compile: bool = False


# --- Root Config ---


class SAM3LoRAConfig(BaseModel):
    """Full SAM3 LoRA training configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "SAM3LoRAConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data or {})

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
