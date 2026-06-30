"""Measurement-only parquet loading and fixed-length cycle tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import META_COLUMNS


@dataclass
class StandardScalerStats:
    mean: torch.Tensor
    scale: torch.Tensor
    channels: list[str]
    eps: float = 1e-6
    clip: float | None = 12.0

    def normalize(self, values: np.ndarray) -> np.ndarray:
        mean = self.mean.cpu().numpy()
        scale = np.maximum(self.scale.cpu().numpy(), self.eps)
        normalized = (values - mean[None, :]) / scale[None, :]
        if self.clip is not None:
            normalized = np.clip(normalized, -float(self.clip), float(self.clip))
        return normalized.astype(np.float32, copy=False)

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.cpu(),
            "scale": self.scale.cpu(),
            "channels": list(self.channels),
            "eps": float(self.eps),
            "clip": self.clip,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "StandardScalerStats":
        return cls(
            mean=state["mean"].float(),
            scale=state["scale"].float(),
            channels=list(state["channels"]),
            eps=float(state["eps"]),
            clip=None if state.get("clip") is None else float(state["clip"]),
        )


@dataclass
class CycleTensors:
    values: torch.Tensor
    sample_ids: torch.Tensor
    anomaly: torch.Tensor
    category: torch.Tensor
    setting: torch.Tensor
    original_lengths: torch.Tensor
    channels: list[str]

    def to_dataset(self) -> "CycleDataset":
        return CycleDataset(self)


class CycleDataset(Dataset):
    def __init__(self, tensors: CycleTensors):
        self.tensors = tensors

    def __len__(self) -> int:
        return int(self.tensors.values.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "values": self.tensors.values[index],
            "sample_ids": self.tensors.sample_ids[index],
            "anomaly": self.tensors.anomaly[index],
            "category": self.tensors.category[index],
            "setting": self.tensors.setting[index],
            "original_lengths": self.tensors.original_lengths[index],
        }


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns, engine="fastparquet")


def infer_measurement_columns(path: str | Path) -> list[str]:
    frame = read_parquet(path)
    channels = [column for column in frame.columns if column not in META_COLUMNS]
    if len(channels) != 58:
        raise RuntimeError(f"Expected 58 measurement columns, found {len(channels)}.")
    return channels


def fit_standard_scaler(
    path: str | Path,
    channels: Sequence[str],
    eps: float,
    clip: float | None,
    limit_samples: int | None = None,
) -> StandardScalerStats:
    columns = list(META_COLUMNS) + list(channels)
    frame = read_parquet(path, columns=columns)
    if limit_samples is not None:
        keep = frame["sample"].drop_duplicates().iloc[:limit_samples]
        frame = frame[frame["sample"].isin(keep)]
    values = frame[list(channels)].to_numpy(dtype=np.float32, copy=False)
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.where(scale > eps, scale, 1.0).astype(np.float32)
    return StandardScalerStats(
        mean=torch.from_numpy(mean),
        scale=torch.from_numpy(scale),
        channels=list(channels),
        eps=eps,
        clip=clip,
    )


def resample_sequence(values: np.ndarray, target_length: int) -> np.ndarray:
    length, channels = values.shape
    if length == target_length:
        return values.astype(np.float32, copy=False)
    if length <= 1:
        return np.repeat(values.astype(np.float32, copy=False), target_length, axis=0)
    old_grid = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
    new_grid = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    out = np.empty((target_length, channels), dtype=np.float32)
    for channel_index in range(channels):
        out[:, channel_index] = np.interp(new_grid, old_grid, values[:, channel_index]).astype(np.float32)
    return out


def _sample_to_arrays(sample_frame: pd.DataFrame, channels: Sequence[str], scaler: StandardScalerStats, target_length: int) -> dict[str, Any]:
    raw = sample_frame[list(channels)].to_numpy(dtype=np.float32, copy=False)
    values = resample_sequence(scaler.normalize(raw), target_length)
    first = sample_frame.iloc[0]
    return {
        "values": values,
        "sample": int(first["sample"]),
        "anomaly": int(bool(first["anomaly"])),
        "category": int(first["category"]),
        "setting": int(first["setting"]),
        "original_length": int(raw.shape[0]),
    }


def load_cycle_tensors(
    path: str | Path,
    channels: Sequence[str],
    scaler: StandardScalerStats,
    target_length: int,
    limit_samples: int | None = None,
) -> CycleTensors:
    columns = list(META_COLUMNS) + list(channels)
    frame = read_parquet(path, columns=columns)
    samples = frame["sample"].drop_duplicates().to_list()
    if limit_samples is not None:
        samples = samples[:limit_samples]
        frame = frame[frame["sample"].isin(samples)]
    rows = [_sample_to_arrays(group, channels, scaler, target_length) for _, group in frame.groupby("sample", sort=False)]
    return CycleTensors(
        values=torch.from_numpy(np.stack([row["values"] for row in rows], axis=0)).float(),
        sample_ids=torch.tensor([row["sample"] for row in rows], dtype=torch.long),
        anomaly=torch.tensor([row["anomaly"] for row in rows], dtype=torch.bool),
        category=torch.tensor([row["category"] for row in rows], dtype=torch.long),
        setting=torch.tensor([row["setting"] for row in rows], dtype=torch.long),
        original_lengths=torch.tensor([row["original_length"] for row in rows], dtype=torch.long),
        channels=list(channels),
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
    )


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
