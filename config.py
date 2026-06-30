"""Configuration and path helpers for TNP-Diffusion v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path("/mnt/disk2/CaiShenghao/ITS/DataDrift")
DATA_DIR = PROJECT_DIR / "data"
SPLITS_DIR = DATA_DIR / "splits"
DRIFTED_DIR = DATA_DIR / "drifted_test_sets"
DEFAULT_OUTPUT_DIR = DATA_DIR / "outputs" / "ours_v1_tnp_diffusion"

DEFAULT_TRAIN_PATH = SPLITS_DIR / "train_normal.parquet"
DEFAULT_VAL_PATH = SPLITS_DIR / "val_normal.parquet"
DEFAULT_CLEAN_TEST_PATH = SPLITS_DIR / "test.parquet"

META_COLUMNS = ("time", "sample", "anomaly", "category", "setting", "action", "active")
SCENARIOS = ("P1", "P2", "P3", "P4", "P5")
SEVERITIES = ("mild", "medium", "severe")


@dataclass
class DataConfig:
    target_length: int = 384
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    scaler_eps: float = 1e-6
    normalize_clip: float | None = None
    limit_train_samples: int | None = None
    limit_eval_samples: int | None = None


@dataclass
class ModelConfig:
    input_channels: int = 58
    target_length: int = 384
    model_dim: int = 160
    depth: int = 5
    heads: int = 5
    ff_mult: int = 4
    dropout: float = 0.05
    beta_min: float = 0.1
    beta_max: float = 20.0
    t_eps: float = 1e-4


@dataclass
class DiffusionConfig:
    beta_min: float = 0.1
    beta_max: float = 20.0
    t_eps: float = 1e-4
    likelihood_steps: int = 24


@dataclass
class ProfileConfig:
    score_probes: int = 8
    score_chunk: int = 4
    profile_steps: int = 14
    profile_lr: float = 0.08
    profile_energy_probes: int = 4
    nuisance_time_steps: int = 8


@dataclass
class TrainConfig:
    seed: int = 177
    epochs: int = 120
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True
    amp_dtype: str = "float16"
    device: str = "cuda"
    log_interval: int = 20
    save_every: int = 20
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    run_name: str = "tnp_diffusion_v1"


@dataclass
class InferenceConfig:
    seed: int = 177
    threshold_q: float = 0.99
    batch_size: int = 16
    amp: bool = True
    amp_dtype: str = "float16"
    device: str = "cuda"
    output_dir: str = str(DEFAULT_OUTPUT_DIR / "inference_q099")
    profile: ProfileConfig = field(default_factory=ProfileConfig)


@dataclass
class ExperimentConfig:
    train_path: str = str(DEFAULT_TRAIN_PATH)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict[str, Any]:
        return to_plain_dict(self)


def to_plain_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_dict(item) for item in value]
    return value


def ensure_dir(path: str | Path) -> Path:
    result = Path(path)
    result.mkdir(parents=True, exist_ok=True)
    return result


def parse_eval_pairs(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=/path/to/file.parquet, got {value}")
        name, path = value.split("=", 1)
        parsed[name] = Path(path)
    return parsed


def q099_mixed_eval_paths(clean_test_path: Path = DEFAULT_CLEAN_TEST_PATH, drifted_dir: Path = DRIFTED_DIR) -> dict[str, Path]:
    paths = {"clean_test": Path(clean_test_path)}
    for scenario in SCENARIOS:
        for severity in SEVERITIES:
            paths[f"{scenario}_{severity}"] = Path(drifted_dir) / f"{scenario}_{severity}_test.parquet"
    return paths
