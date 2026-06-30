"""Configuration for CCSD-QKV v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path("/mnt/disk2/CaiShenghao/ITS/DataDrift")
DATA_DIR = PROJECT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
Q099_DRIFTED_DIR = DATA_DIR / "drifted_test_sets"

DEFAULT_TRAIN_PATH = SPLITS_DIR / "train_normal.parquet"
DEFAULT_VAL_PATH = SPLITS_DIR / "val_normal.parquet"
DEFAULT_CLEAN_TEST_PATH = SPLITS_DIR / "test.parquet"
DEFAULT_OUTPUT_DIR = DATA_DIR / "outputs" / "ours_v1_ccsd_qkv"

META_COLUMNS = ("time", "sample", "anomaly", "category", "setting", "action", "active")
SEVERITIES = ("mild", "medium", "severe")
SCENARIOS = ("P1", "P2", "P3", "P4", "P5")


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""

    target_length: int = 512
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    scaler_eps: float = 1e-6
    normalize_clip: float | None = 12.0
    limit_train_samples: int | None = None
    limit_eval_samples: int | None = None


@dataclass
class ModelConfig:
    """Transformer denoiser configuration."""

    input_channels: int = 58
    target_length: int = 512
    model_dim: int = 128
    depth: int = 4
    heads: int = 4
    ff_mult: int = 4
    sigma_embed_dim: int = 128
    dropout: float = 0.05


@dataclass
class DiffusionConfig:
    """Continuous-noise conditional denoising configuration."""

    sigma_min: float = 0.02
    sigma_max: float = 1.0
    train_mask_ratio: float = 0.25
    eval_mask_ratio: float = 0.25
    mask_strategy: str = "mixed"
    element_mask_prob: float = 0.50
    time_block_mask_prob: float = 0.25
    patch_mask_prob: float = 0.25
    min_time_block: int = 16
    max_time_block: int = 96
    patch_channel_fraction: float = 0.35


@dataclass
class TrainConfig:
    """Optimization configuration."""

    seed: int = 177
    epochs: int = 120
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amp: bool = True
    amp_dtype: str = "float16"
    device: str = "cuda"
    ema_decay: float = 0.999
    log_interval: int = 20
    save_every: int = 20
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    run_name: str = "ccsd_qkv_v1"


@dataclass
class InferenceConfig:
    """Inference and evaluation configuration."""

    seed: int = 177
    batch_size: int = 32
    mc_samples: int = 32
    mc_chunk: int = 4
    threshold_q: float = 0.99
    amp: bool = True
    amp_dtype: str = "float16"
    device: str = "cuda"
    output_dir: str = str(DEFAULT_OUTPUT_DIR / "inference")


@dataclass
class ExperimentConfig:
    """Serializable experiment configuration saved in checkpoints."""

    train_path: str = str(DEFAULT_TRAIN_PATH)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


def to_plain_dict(obj: Any) -> Any:
    """Converts nested dataclasses and paths into JSON-friendly objects."""
    if is_dataclass(obj):
        return {key: to_plain_dict(value) for key, value in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): to_plain_dict(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_dict(value) for value in obj]
    return obj


def ensure_dir(path: str | Path) -> Path:
    """Creates and returns a directory path."""
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def parse_eval_pairs(values: list[str]) -> dict[str, Path]:
    """Parses CLI values of the form NAME=/path/to/file.parquet."""
    parsed: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Expected NAME=/path/to/file.parquet, got: {item}")
        name, path = item.split("=", 1)
        if not name:
            raise ValueError(f"Empty dataset name in eval argument: {item}")
        parsed[name] = Path(path)
    return parsed


def q099_eval_paths(clean_test_path: Path = DEFAULT_CLEAN_TEST_PATH, drifted_dir: Path = Q099_DRIFTED_DIR) -> dict[str, Path]:
    """Builds standard q=0.99 clean and drifted mixed-test eval paths."""
    paths: dict[str, Path] = {"clean_test": Path(clean_test_path)}
    for scenario in SCENARIOS:
        for severity in SEVERITIES:
            paths[f"{scenario}_{severity}"] = Path(drifted_dir) / f"{scenario}_{severity}_test.parquet"
    return paths
