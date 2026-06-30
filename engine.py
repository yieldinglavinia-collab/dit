"""Training, profile scoring, and evaluation utilities for TNP-Diffusion."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import nn
from tqdm.auto import tqdm

from config import DiffusionConfig, InferenceConfig, ProfileConfig, TrainConfig
from data import move_to_device
from model import ScoreTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def resolve_amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name.lower() == "bfloat16" else torch.float16


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        state = model.state_dict()
        for key, shadow in self.shadow.items():
            shadow.mul_(self.decay).add_(state[key].detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.shadow.items()}

    @staticmethod
    def apply_to(model: nn.Module, ema_state: dict[str, torch.Tensor]) -> None:
        state = model.state_dict()
        for key, value in ema_state.items():
            if key in state:
                state[key].copy_(value.to(state[key].device))


class VPSchedule:
    """Continuous VP-SDE closed-form perturbation schedule."""

    def __init__(self, cfg: DiffusionConfig):
        self.cfg = cfg

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.cfg.beta_min + t * (self.cfg.beta_max - self.cfg.beta_min)

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.cfg.beta_min * t + 0.5 * (self.cfg.beta_max - self.cfg.beta_min) * t.pow(2)

    def alpha_sigma(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.exp(-0.5 * self.integral_beta(t))
        sigma = torch.sqrt(torch.clamp(1.0 - alpha.pow(2), min=1e-8))
        return alpha, sigma

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(t)
        return alpha.view(-1, 1, 1) * x0 + sigma.view(-1, 1, 1) * noise

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        alpha, sigma = self.alpha_sigma(t)
        return (x_t - sigma.view(-1, 1, 1) * eps_pred) / alpha.view(-1, 1, 1).clamp_min(1e-5)


def sample_train_time(batch: int, cfg: DiffusionConfig, device: torch.device) -> torch.Tensor:
    if cfg.train_time_sampling == "uniform":
        return torch.empty(batch, device=device).uniform_(cfg.t_eps, 1.0)
    # logSNR-ish emphasis: sample uniform t but mix in endpoints.
    u = torch.rand(batch, device=device)
    t = torch.sigmoid(torch.empty(batch, device=device).uniform_(-5.0, 5.0))
    return torch.where(u < 0.75, t, torch.empty(batch, device=device).uniform_(cfg.t_eps, 1.0)).clamp(cfg.t_eps, 1.0)


def dsm_loss(model: ScoreTransformer, values: torch.Tensor, schedule: VPSchedule) -> tuple[torch.Tensor, dict[str, float]]:
    batch = values.shape[0]
    t = sample_train_time(batch, schedule.cfg, values.device)
    noise = torch.randn_like(values)
    x_t = schedule.q_sample(values, t, noise)
    eps_pred = model(x_t, t)
    per_sample = (eps_pred - noise).pow(2).flatten(1).mean(dim=1)
    loss = per_sample.mean()
    return loss, {
        "loss": float(loss.detach().item()),
        "t_mean": float(t.detach().mean().item()),
        "eps_mse": float(per_sample.detach().mean().item()),
    }


def train_one_epoch(
    model: ScoreTransformer,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    schedule: VPSchedule,
    train_cfg: TrainConfig,
    device: torch.device,
    ema: ModelEMA | None,
    epoch: int,
) -> dict[str, float]:
    model.train()
    scaler = torch.amp.GradScaler(device=device.type, enabled=train_cfg.amp and device.type == "cuda")
    amp_dtype = resolve_amp_dtype(train_cfg.amp_dtype)
    sums = {"loss": 0.0, "t_mean": 0.0, "eps_mse": 0.0}
    steps = 0
    progress = tqdm(loader, desc=f"train:{epoch:03d}", leave=False)
    for step, batch in enumerate(progress):
        steps += 1
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=train_cfg.amp and device.type == "cuda"):
            loss, loss_metrics = dsm_loss(model, batch["values"], schedule)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        for key in sums:
            sums[key] += loss_metrics[key]
        if step % train_cfg.log_interval == 0:
            progress.set_postfix(loss=f"{sums['loss'] / steps:.4f}", t=f"{sums['t_mean'] / steps:.3f}")
    return {key: value / max(steps, 1) for key, value in sums.items()}


def fixed_score_times(probes: int, device: torch.device, cfg: DiffusionConfig) -> torch.Tensor:
    if probes <= 1:
        return torch.tensor([0.5], device=device)
    return torch.linspace(max(cfg.t_eps, 0.03), 0.95, steps=probes, device=device)


def static_energy_with_fixed_noise(
    model: ScoreTransformer,
    z: torch.Tensor,
    schedule: VPSchedule,
    t_values: torch.Tensor,
    noises: torch.Tensor,
) -> torch.Tensor:
    """Differentiable denoising energy for a fixed [K, B, T, C] noise bank."""
    energies = []
    for index, t_scalar in enumerate(t_values):
        t = torch.full((z.shape[0],), float(t_scalar.item()), device=z.device)
        noise = noises[index]
        x_t = schedule.q_sample(z, t, noise)
        eps_pred = model(x_t, t)
        energies.append((eps_pred - noise).pow(2).flatten(1).mean(dim=1))
    return torch.stack(energies, dim=0).mean(dim=0)


@torch.no_grad()
def static_energy(
    model: ScoreTransformer,
    values: torch.Tensor,
    schedule: VPSchedule,
    profile_cfg: ProfileConfig,
    amp: bool,
    amp_dtype: str,
) -> torch.Tensor:
    """MC denoising energy proxy for clean diffusion normality."""
    device = values.device
    dtype = resolve_amp_dtype(amp_dtype)
    times = fixed_score_times(profile_cfg.score_probes, device, schedule.cfg)
    total = torch.zeros(values.shape[0], device=device)
    done = 0
    for start in range(0, len(times), profile_cfg.score_chunk):
        subset = times[start : start + profile_cfg.score_chunk]
        chunk_sum = torch.zeros(values.shape[0], device=device)
        for t_scalar in subset:
            t = torch.full((values.shape[0],), float(t_scalar.item()), device=device)
            noise = torch.randn_like(values)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp and device.type == "cuda"):
                x_t = schedule.q_sample(values, t, noise)
                eps_pred = model(x_t, t)
                chunk_sum += (eps_pred - noise).pow(2).flatten(1).mean(dim=1)
        total += chunk_sum
        done += len(subset)
    return total / max(done, 1)


@torch.no_grad()
def posterior_variance_diag(
    model: ScoreTransformer,
    values: torch.Tensor,
    schedule: VPSchedule,
    profile_cfg: ProfileConfig,
    amp: bool,
    amp_dtype: str,
) -> torch.Tensor:
    """Diffusion-induced diagonal posterior uncertainty via x0-hat variance."""
    device = values.device
    dtype = resolve_amp_dtype(amp_dtype)
    x0_hats = []
    probes = int(profile_cfg.posterior_probes)
    for start in range(0, probes, profile_cfg.posterior_chunk):
        count = min(profile_cfg.posterior_chunk, probes - start)
        v_rep = values.repeat_interleave(count, dim=0)
        t = torch.empty(v_rep.shape[0], device=device).uniform_(profile_cfg.posterior_t_min, profile_cfg.posterior_t_max)
        noise = torch.randn_like(v_rep)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp and device.type == "cuda"):
            x_t = schedule.q_sample(v_rep, t, noise)
            eps_pred = model(x_t, t)
            x0_hat = schedule.predict_x0(x_t, t, eps_pred)
        x0_hats.append(x0_hat.float().view(values.shape[0], count, *values.shape[1:]))
    stacked = torch.cat(x0_hats, dim=1)
    var = stacked.var(dim=1, unbiased=False)
    return (profile_cfg.nuisance_var_scale * var + profile_cfg.nuisance_var_floor).detach().clamp_min(1e-6)


def nuisance_code(delta: torch.Tensor, var_diag: torch.Tensor) -> torch.Tensor:
    """Gaussian diagonal nuisance negative log code length, averaged per cell."""
    return 0.5 * (delta.pow(2) / var_diag + torch.log(var_diag)).flatten(1).mean(dim=1)


def profile_score_batch(
    model: ScoreTransformer,
    y: torch.Tensor,
    schedule: VPSchedule,
    inference_cfg: InferenceConfig,
) -> dict[str, torch.Tensor]:
    """Computes TNP profiled score for one batch."""
    profile_cfg = inference_cfg.profile
    if profile_cfg.profile_score_mode == "static":
        clean_e = static_energy(model, y, schedule, profile_cfg, inference_cfg.amp, inference_cfg.amp_dtype)
        zeros = torch.zeros_like(clean_e)
        return {
            "score": clean_e,
            "static_energy": clean_e,
            "profiled_energy": clean_e,
            "nuisance_code": zeros,
            "correction_rms": zeros,
            "profile_improvement": zeros,
        }

    var_diag = posterior_variance_diag(model, y, schedule, profile_cfg, inference_cfg.amp, inference_cfg.amp_dtype)
    z = y.detach().clone().requires_grad_(True)
    t_values = fixed_score_times(profile_cfg.profile_energy_probes, y.device, schedule.cfg)
    noise_bank = torch.randn(len(t_values), *y.shape, device=y.device)
    optimizer = torch.optim.Adam([z], lr=profile_cfg.profile_lr)

    progress = range(profile_cfg.profile_steps)
    for _ in progress:
        optimizer.zero_grad(set_to_none=True)
        clean_e = static_energy_with_fixed_noise(model, z, schedule, t_values, noise_bank)
        code = nuisance_code(y - z, var_diag)
        objective = (clean_e + code).mean()
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            z.clamp_(-profile_cfg.profile_clip, profile_cfg.profile_clip)

    with torch.no_grad():
        profiled_e = static_energy(model, z.detach(), schedule, profile_cfg, inference_cfg.amp, inference_cfg.amp_dtype)
        static_e = static_energy(model, y, schedule, profile_cfg, inference_cfg.amp, inference_cfg.amp_dtype)
        code = nuisance_code(y - z.detach(), var_diag)
        score = profiled_e + code
        correction = y - z.detach()
        correction_rms = correction.pow(2).flatten(1).mean(dim=1).sqrt()
    return {
        "score": score,
        "static_energy": static_e,
        "profiled_energy": profiled_e,
        "nuisance_code": code,
        "correction_rms": correction_rms,
        "profile_improvement": static_e - score,
    }


@torch.no_grad()
def _empty_cache_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def score_dataloader(
    model: ScoreTransformer,
    loader: Iterable[dict[str, torch.Tensor]],
    schedule: VPSchedule,
    inference_cfg: InferenceConfig,
    device: torch.device,
    dataset_name: str,
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    progress = tqdm(loader, desc=f"score:{dataset_name}", leave=False)
    for batch in progress:
        batch = move_to_device(batch, device)
        result = profile_score_batch(model, batch["values"], schedule, inference_cfg)
        for index in range(batch["values"].shape[0]):
            rows.append(
                {
                    "split": dataset_name,
                    "sample": int(batch["sample_ids"][index].item()),
                    "anomaly": bool(batch["anomaly"][index].item()),
                    "category": int(batch["category"][index].item()),
                    "setting": int(batch["setting"][index].item()),
                    "original_length": int(batch["original_lengths"][index].item()),
                    "score": float(result["score"][index].detach().cpu().item()),
                    "static_energy": float(result["static_energy"][index].detach().cpu().item()),
                    "profiled_energy": float(result["profiled_energy"][index].detach().cpu().item()),
                    "nuisance_code": float(result["nuisance_code"][index].detach().cpu().item()),
                    "correction_rms": float(result["correction_rms"][index].detach().cpu().item()),
                    "profile_improvement": float(result["profile_improvement"][index].detach().cpu().item()),
                }
            )
        _empty_cache_if_cuda(device)
    return pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)


def metrics_at_tau(scores: pd.DataFrame, tau: float) -> dict[str, Any]:
    labels = scores["anomaly"].astype(bool)
    alarms = scores["score"].to_numpy(dtype=np.float64) > float(tau)
    result: dict[str, Any] = {
        "n_samples": int(len(scores)),
        "tau": float(tau),
        "n_normal": int((~labels).sum()),
        "n_anomaly": int(labels.sum()),
        "score_mean": float(scores["score"].mean()),
        "score_median": float(scores["score"].median()),
        "score_p95": float(scores["score"].quantile(0.95)),
        "score_p99": float(scores["score"].quantile(0.99)),
    }
    normal_mask = (~labels).to_numpy()
    anomaly_mask = labels.to_numpy()
    if normal_mask.any():
        result["FAR"] = float(alarms[normal_mask].mean())
        result["false_alarm_count"] = int(alarms[normal_mask].sum())
    if anomaly_mask.any():
        result["TPR"] = float(alarms[anomaly_mask].mean())
        result["true_positive_count"] = int(alarms[anomaly_mask].sum())
    if labels.nunique() >= 2:
        y_true = labels.astype(int).to_numpy()
        y_score = scores["score"].to_numpy(dtype=np.float64)
        result["AUROC"] = float(metrics.roc_auc_score(y_true, y_score))
        result["AP"] = float(metrics.average_precision_score(y_true, y_score))
    return result
