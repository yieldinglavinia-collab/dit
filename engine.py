"""Training and scoring utilities for CCSD-QKV."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch import nn
from tqdm.auto import tqdm

from config import DiffusionConfig, InferenceConfig, TrainConfig
from data import move_batch_to_device
from model import CCSDQKVDenoiser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name.lower() == "bfloat16":
        return torch.bfloat16
    return torch.float16


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    return value


class ModelEMA:
    """Simple exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.state_dict().items()
            if torch.is_floating_point(parameter)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, shadow_value in self.shadow.items():
            shadow_value.mul_(self.decay).add_(state[name].detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.shadow.items()}

    def copy_to(self, model: nn.Module) -> None:
        state = model.state_dict()
        for name, value in self.shadow.items():
            if name in state:
                state[name].copy_(value.to(state[name].device))

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor], decay: float) -> "ModelEMA":
        obj = cls.__new__(cls)
        obj.decay = float(decay)
        obj.shadow = {name: value.detach().clone() for name, value in state.items()}
        return obj


def sample_sigmas(batch_size: int, cfg: DiffusionConfig, device: torch.device) -> torch.Tensor:
    """Samples log-uniform sigma values."""
    log_min = math.log(cfg.sigma_min)
    log_max = math.log(cfg.sigma_max)
    return torch.exp(torch.empty(batch_size, device=device).uniform_(log_min, log_max))


def _ensure_nonempty_mask(mask: torch.Tensor) -> torch.Tensor:
    batch, time_steps, channels = mask.shape
    flat = mask.view(batch, -1)
    empty = flat.sum(dim=1) < 0.5
    if empty.any():
        choices = torch.randint(0, time_steps * channels, (int(empty.sum().item()),), device=mask.device)
        flat[empty, choices] = 1.0
    return flat.view(batch, time_steps, channels)


def _element_mask(batch: int, time_steps: int, channels: int, ratio: float, device: torch.device) -> torch.Tensor:
    return (torch.rand(batch, time_steps, channels, device=device) < ratio).float()


def _block_or_patch_mask(
    batch: int,
    time_steps: int,
    channels: int,
    ratio: float,
    cfg: DiffusionConfig,
    device: torch.device,
    patch: bool,
) -> torch.Tensor:
    mask = torch.zeros(batch, time_steps, channels, device=device)
    target_cells = max(1, int(round(time_steps * channels * ratio)))
    min_block = max(1, min(cfg.min_time_block, time_steps))
    max_block = max(min_block, min(cfg.max_time_block, time_steps))
    patch_channels = max(1, int(round(channels * cfg.patch_channel_fraction)))
    for batch_index in range(batch):
        changed = 0
        attempts = 0
        while changed < target_cells and attempts < 128:
            attempts += 1
            block_len = int(torch.randint(min_block, max_block + 1, (1,), device=device).item())
            start = int(torch.randint(0, max(time_steps - block_len + 1, 1), (1,), device=device).item())
            if patch:
                selected = torch.randperm(channels, device=device)[:patch_channels]
                mask[batch_index, start : start + block_len, selected] = 1.0
            else:
                mask[batch_index, start : start + block_len, :] = 1.0
            changed = int(mask[batch_index].sum().item())
    return mask


def sample_target_mask(
    batch: int,
    time_steps: int,
    channels: int,
    ratio: float,
    cfg: DiffusionConfig,
    device: torch.device,
) -> torch.Tensor:
    """Samples internal target masks, independent of anomaly/drift labels."""
    strategy = cfg.mask_strategy.lower()
    if strategy == "element":
        mask = _element_mask(batch, time_steps, channels, ratio, device)
    elif strategy == "time_block":
        mask = _block_or_patch_mask(batch, time_steps, channels, ratio, cfg, device, patch=False)
    elif strategy == "patch":
        mask = _block_or_patch_mask(batch, time_steps, channels, ratio, cfg, device, patch=True)
    elif strategy == "mixed":
        probs = torch.tensor(
            [cfg.element_mask_prob, cfg.time_block_mask_prob, cfg.patch_mask_prob],
            device=device,
            dtype=torch.float32,
        )
        probs = probs / probs.sum().clamp_min(1e-6)
        choices = torch.multinomial(probs, batch, replacement=True)
        mask = torch.zeros(batch, time_steps, channels, device=device)
        for choice in range(3):
            selected = choices == choice
            count = int(selected.sum().item())
            if count == 0:
                continue
            if choice == 0:
                part = _element_mask(count, time_steps, channels, ratio, device)
            elif choice == 1:
                part = _block_or_patch_mask(count, time_steps, channels, ratio, cfg, device, patch=False)
            else:
                part = _block_or_patch_mask(count, time_steps, channels, ratio, cfg, device, patch=True)
            mask[selected] = part
    else:
        raise ValueError(f"Unknown mask strategy: {cfg.mask_strategy}")
    return _ensure_nonempty_mask(mask)


def masked_denoising_loss(
    model: CCSDQKVDenoiser,
    values: torch.Tensor,
    diffusion_cfg: DiffusionConfig,
    mask_ratio: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    batch, time_steps, channels = values.shape
    device = values.device
    mask = sample_target_mask(batch, time_steps, channels, mask_ratio, diffusion_cfg, device)
    sigma = sample_sigmas(batch, diffusion_cfg, device)
    noise = torch.randn_like(values)
    noised = values + sigma.view(batch, 1, 1) * noise
    denoiser_input = values * (1.0 - mask) + noised * mask
    prediction = model(denoiser_input, mask, sigma)
    per_sample_sum = ((prediction - noise).pow(2) * mask).flatten(1).sum(dim=1)
    per_sample_count = mask.flatten(1).sum(dim=1).clamp_min(1.0)
    per_sample = per_sample_sum / per_sample_count
    loss = per_sample.mean()
    return loss, {
        "loss": float(loss.detach().item()),
        "mask_rate": float(mask.mean().detach().item()),
        "sigma_mean": float(sigma.mean().detach().item()),
    }


def train_one_epoch(
    model: CCSDQKVDenoiser,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    diffusion_cfg: DiffusionConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    ema: ModelEMA | None,
    epoch: int,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler(device=device.type, enabled=train_cfg.amp and device.type == "cuda")
    amp_dtype = resolve_amp_dtype(train_cfg.amp_dtype)
    sums = {"loss": 0.0, "mask_rate": 0.0, "sigma_mean": 0.0}
    batches = 0
    progress = tqdm(loader, desc=f"train:{epoch:03d}", leave=False)
    for step, batch in enumerate(progress):
        batches += 1
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=train_cfg.amp and device.type == "cuda"):
            loss, metrics_dict = masked_denoising_loss(
                model=model,
                values=batch["values"],
                diffusion_cfg=diffusion_cfg,
                mask_ratio=diffusion_cfg.train_mask_ratio,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        for key in sums:
            sums[key] += metrics_dict[key]
        if step % train_cfg.log_interval == 0:
            progress.set_postfix(loss=f"{sums['loss'] / batches:.4f}", mask=f"{sums['mask_rate'] / batches:.3f}")
    return {key: value / max(batches, 1) for key, value in sums.items()}


@torch.no_grad()
def score_batch(
    model: CCSDQKVDenoiser,
    values: torch.Tensor,
    diffusion_cfg: DiffusionConfig,
    inference_cfg: InferenceConfig,
) -> torch.Tensor:
    """Computes Monte Carlo averaged conditional denoising risk for a batch."""
    model.eval()
    batch, time_steps, channels = values.shape
    device = values.device
    remaining = int(inference_cfg.mc_samples)
    total = torch.zeros(batch, device=device)
    completed = 0
    while remaining > 0:
        repeats = min(int(inference_cfg.mc_chunk), remaining)
        x_rep = values.repeat_interleave(repeats, dim=0)
        mask = sample_target_mask(
            batch * repeats,
            time_steps,
            channels,
            diffusion_cfg.eval_mask_ratio,
            diffusion_cfg,
            device,
        )
        sigma = sample_sigmas(batch * repeats, diffusion_cfg, device)
        noise = torch.randn_like(x_rep)
        noised = x_rep + sigma.view(batch * repeats, 1, 1) * noise
        denoiser_input = x_rep * (1.0 - mask) + noised * mask
        prediction = model(denoiser_input, mask, sigma)
        per_rep = ((prediction - noise).pow(2) * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1.0)
        total += per_rep.view(batch, repeats).sum(dim=1)
        completed += repeats
        remaining -= repeats
    return total / max(completed, 1)


@torch.no_grad()
def score_dataloader(
    model: CCSDQKVDenoiser,
    loader: Iterable[dict[str, torch.Tensor]],
    diffusion_cfg: DiffusionConfig,
    inference_cfg: InferenceConfig,
    device: torch.device,
    dataset_name: str,
) -> pd.DataFrame:
    """Scores a dataloader and returns one score per cycle."""
    amp_dtype = resolve_amp_dtype(inference_cfg.amp_dtype)
    rows: list[dict[str, Any]] = []
    progress = tqdm(loader, desc=f"score:{dataset_name}", leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=inference_cfg.amp and device.type == "cuda"):
            scores = score_batch(model, batch["values"], diffusion_cfg, inference_cfg)
        for index in range(scores.shape[0]):
            rows.append(
                {
                    "split": dataset_name,
                    "sample": int(batch["sample_ids"][index].item()),
                    "score": float(scores[index].detach().cpu().item()),
                    "anomaly": bool(batch["anomaly"][index].item()),
                    "category": int(batch["category"][index].item()),
                    "setting": int(batch["setting"][index].item()),
                    "original_length": int(batch["original_lengths"][index].item()),
                }
            )
    return pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)


def metrics_at_threshold(scores: pd.DataFrame, tau: float) -> dict[str, Any]:
    """Computes FAR/TPR/AUROC/AP where labels are available."""
    result: dict[str, Any] = {
        "n_samples": int(len(scores)),
        "tau": float(tau),
    }
    if scores.empty:
        return result
    labels = scores["anomaly"].astype(bool)
    alarms = scores["score"].to_numpy(dtype=np.float64) > float(tau)
    normal = ~labels
    anomaly = labels
    result["n_normal"] = int(normal.sum())
    result["n_anomaly"] = int(anomaly.sum())
    if normal.any():
        result["FAR"] = float(alarms[normal.to_numpy()].mean())
        result["false_alarm_count"] = int(alarms[normal.to_numpy()].sum())
    if anomaly.any():
        result["TPR"] = float(alarms[anomaly.to_numpy()].mean())
        result["true_positive_count"] = int(alarms[anomaly.to_numpy()].sum())
    if labels.nunique() >= 2:
        y_true = labels.astype(int).to_numpy()
        y_score = scores["score"].to_numpy(dtype=np.float64)
        result["AUROC"] = float(metrics.roc_auc_score(y_true, y_score))
        result["AP"] = float(metrics.average_precision_score(y_true, y_score))
    result["score_mean"] = float(scores["score"].mean())
    result["score_median"] = float(scores["score"].median())
    result["score_p95"] = float(scores["score"].quantile(0.95))
    result["score_p99"] = float(scores["score"].quantile(0.99))
    return result
