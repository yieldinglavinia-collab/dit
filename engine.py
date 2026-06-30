"""Training, profile scoring, and evaluation utilities for TNP-Diffusion."""

from __future__ import annotations

import json
import math
import random
from contextlib import contextmanager
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


@contextmanager
def disabled_parameter_grads(model: nn.Module):
    """Disable parameter gradients while preserving input gradients."""
    parameters = list(model.parameters())
    states = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(parameters, states):
            parameter.requires_grad_(state)


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


def _score_from_eps(model: ScoreTransformer, x_t: torch.Tensor, t: torch.Tensor, schedule: VPSchedule) -> torch.Tensor:
    _, sigma = schedule.alpha_sigma(t)
    eps_pred = model(x_t, t)
    return -eps_pred / sigma.view(-1, 1, 1).clamp_min(1e-5)


def _probability_flow_velocity(
    model: ScoreTransformer,
    x_t: torch.Tensor,
    t: torch.Tensor,
    schedule: VPSchedule,
) -> torch.Tensor:
    beta = schedule.beta(t).view(-1, 1, 1)
    score = _score_from_eps(model, x_t, t, schedule)
    return -0.5 * beta * x_t - 0.5 * beta * score


def _divergence_hutchinson(
    velocity: torch.Tensor,
    x_t: torch.Tensor,
    probes: int,
) -> torch.Tensor:
    estimates = []
    for _ in range(max(int(probes), 1)):
        noise = torch.empty_like(x_t).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        projected = (velocity * noise).flatten(1).sum()
        grad = torch.autograd.grad(projected, x_t, create_graph=True, retain_graph=True)[0]
        estimates.append((grad * noise).flatten(1).sum(dim=1))
    return torch.stack(estimates, dim=0).mean(dim=0)


def probability_flow_nll(
    model: ScoreTransformer,
    values: torch.Tensor,
    schedule: VPSchedule,
    diffusion_cfg: DiffusionConfig,
) -> torch.Tensor:
    """Per-cell negative log likelihood from the VP probability-flow ODE.

    The ODE is integrated from t=t_eps to t=1, and the change-of-variables
    divergence is estimated with Hutchinson probes. This implements the README
    clean-normal anchor E_theta(x) = -log p_theta(x), rather than using the
    DSM training loss itself as an anomaly score.
    """
    steps = max(int(diffusion_cfg.likelihood_steps), 1)
    x = values
    div_integral = torch.zeros(values.shape[0], device=values.device, dtype=values.dtype)
    t0 = float(diffusion_cfg.t_eps)
    dt = (1.0 - t0) / steps
    for step in range(steps):
        t_scalar = t0 + (step + 0.5) * dt
        t = torch.full((values.shape[0],), t_scalar, device=values.device, dtype=values.dtype)
        x_req = x.requires_grad_(True)
        velocity = _probability_flow_velocity(model, x_req, t, schedule)
        div = _divergence_hutchinson(velocity, x_req, diffusion_cfg.likelihood_hutchinson_probes)
        div_integral = div_integral + div * dt
        x = x_req + velocity * dt
    log_p1 = -0.5 * (x.pow(2) + math.log(2.0 * math.pi)).flatten(1).sum(dim=1)
    nll = -(log_p1 + div_integral)
    return nll / values[0].numel()


def clean_energy_for_profile(
    model: ScoreTransformer,
    z: torch.Tensor,
    schedule: VPSchedule,
) -> torch.Tensor:
    """Differentiable clean-normal energy E_theta(z) used inside profiling."""
    return probability_flow_nll(model, z, schedule, schedule.cfg)


@torch.no_grad()
def static_energy(
    model: ScoreTransformer,
    values: torch.Tensor,
    schedule: VPSchedule,
    profile_cfg: ProfileConfig,
    amp: bool,
    amp_dtype: str,
) -> torch.Tensor:
    """Probability-flow ODE clean-normal negative log likelihood."""
    dtype = resolve_amp_dtype(amp_dtype)
    with torch.enable_grad(), torch.autocast(device_type=values.device.type, dtype=dtype, enabled=amp and values.device.type == "cuda"):
        return probability_flow_nll(model, values, schedule, schedule.cfg).detach()


def posterior_variance_diag(
    model: ScoreTransformer,
    values: torch.Tensor,
    schedule: VPSchedule,
    profile_cfg: ProfileConfig,
    amp: bool,
    amp_dtype: str,
    time_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Diffusion-induced diagonal posterior uncertainty via x0-hat variance."""
    device = values.device
    dtype = resolve_amp_dtype(amp_dtype)
    x0_hats = []
    if time_values is None:
        probes = int(profile_cfg.posterior_probes)
        time_values = torch.empty(probes, device=device).uniform_(profile_cfg.posterior_t_min, profile_cfg.posterior_t_max)
    probes = int(time_values.numel())
    for start in range(0, probes, profile_cfg.posterior_chunk):
        count = min(profile_cfg.posterior_chunk, probes - start)
        v_rep = values.repeat_interleave(count, dim=0)
        t = time_values[start : start + count].repeat(values.shape[0])
        noise = torch.randn_like(v_rep)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp and device.type == "cuda"):
            x_t = schedule.q_sample(v_rep, t, noise)
            eps_pred = model(x_t, t)
            x0_hat = schedule.predict_x0(x_t, t, eps_pred)
        x0_hats.append(x0_hat.float().view(values.shape[0], count, *values.shape[1:]))
    stacked = torch.cat(x0_hats, dim=1)
    var = stacked.var(dim=1, unbiased=False)
    return (profile_cfg.nuisance_var_scale * var + profile_cfg.nuisance_var_floor).clamp_min(1e-6)


def nuisance_code(delta: torch.Tensor, var_diag: torch.Tensor) -> torch.Tensor:
    """Gaussian diagonal nuisance negative log code length, averaged per cell."""
    return 0.5 * (delta.pow(2) / var_diag + torch.log(var_diag) + math.log(2.0 * math.pi)).flatten(1).mean(dim=1)


def nuisance_time_grid(profile_cfg: ProfileConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    components = max(int(profile_cfg.nuisance_time_components), 1)
    if components == 1:
        mid = 0.5 * (profile_cfg.posterior_t_min + profile_cfg.posterior_t_max)
        return torch.tensor([mid], device=device, dtype=dtype)
    return torch.linspace(profile_cfg.posterior_t_min, profile_cfg.posterior_t_max, steps=components, device=device, dtype=dtype)


def nuisance_mixture_code(
    model: ScoreTransformer,
    delta: torch.Tensor,
    z: torch.Tensor,
    schedule: VPSchedule,
    profile_cfg: ProfileConfig,
    amp: bool,
    amp_dtype: str,
) -> torch.Tensor:
    """Time-mixture nuisance code -log int N(delta; 0, K_theta(z,t)) pi(t) dt.

    K_theta is represented by a diagonal covariance estimated from local
    denoising posterior samples at each nuisance time. The mixture normalization
    is kept with logsumexp, so large correction scale is not free.
    """
    component_times = nuisance_time_grid(profile_cfg, z.device, z.dtype)
    component_codes = []
    noise_probes = max(int(profile_cfg.nuisance_noise_probes), 1)
    for t_scalar in component_times:
        local_times = torch.full((noise_probes,), float(t_scalar.item()), device=z.device, dtype=z.dtype)
        var_diag = posterior_variance_diag(model, z, schedule, profile_cfg, amp, amp_dtype, time_values=local_times)
        component_codes.append(nuisance_code(delta, var_diag))
    stacked_codes = torch.stack(component_codes, dim=0)
    cells = float(delta[0].numel())
    stacked_nll = stacked_codes * cells
    mixture_nll = -torch.logsumexp(-stacked_nll, dim=0) + math.log(float(stacked_nll.shape[0]))
    return mixture_nll / cells


def initialize_profile_latents(y: torch.Tensor, profile_cfg: ProfileConfig) -> tuple[torch.Tensor, int]:
    starts = max(int(profile_cfg.profile_starts), 1)
    if starts == 1:
        return y.detach().clone().requires_grad_(True), starts
    seeds = [y.detach()]
    for _ in range(starts - 1):
        seeds.append(y.detach() + profile_cfg.profile_start_scale * torch.randn_like(y))
    z = torch.cat(seeds, dim=0).requires_grad_(True)
    return z, starts


def repeat_for_profile_starts(y: torch.Tensor, starts: int) -> torch.Tensor:
    if starts == 1:
        return y
    return y.repeat(starts, 1, 1)


def select_best_profile(
    values: torch.Tensor,
    starts: int,
    score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if starts == 1:
        return values, score
    batch = score.shape[0] // starts
    score_matrix = score.view(starts, batch).transpose(0, 1)
    best = score_matrix.argmin(dim=1)
    values_view = values.view(starts, batch, *values.shape[1:]).transpose(0, 1)
    chosen = values_view[torch.arange(batch, device=values.device), best]
    chosen_score = score_matrix[torch.arange(batch, device=score.device), best]
    return chosen, chosen_score


def profile_score_batch(
    model: ScoreTransformer,
    y: torch.Tensor,
    schedule: VPSchedule,
    inference_cfg: InferenceConfig,
) -> dict[str, torch.Tensor]:
    with disabled_parameter_grads(model):
        return _profile_score_batch_impl(model, y, schedule, inference_cfg)


def _profile_score_batch_impl(
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

    z, starts = initialize_profile_latents(y, profile_cfg)
    y_profile = repeat_for_profile_starts(y, starts)
    optimizer = torch.optim.Adam([z], lr=profile_cfg.profile_lr)

    progress = range(profile_cfg.profile_steps)
    for _ in progress:
        optimizer.zero_grad(set_to_none=True)
        clean_e = clean_energy_for_profile(model, z, schedule)
        code = nuisance_mixture_code(
            model,
            y_profile - z,
            z,
            schedule,
            profile_cfg,
            inference_cfg.amp,
            inference_cfg.amp_dtype,
        )
        objective = (clean_e + code).mean()
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            z.clamp_(-profile_cfg.profile_clip, profile_cfg.profile_clip)

    with torch.no_grad():
        z_all = z.detach()
        profiled_e_all = static_energy(model, z_all, schedule, profile_cfg, inference_cfg.amp, inference_cfg.amp_dtype)
        code_all = nuisance_mixture_code(
            model,
            y_profile - z_all,
            z_all,
            schedule,
            profile_cfg,
            inference_cfg.amp,
            inference_cfg.amp_dtype,
        )
        score_all = profiled_e_all + code_all
        z_best, score = select_best_profile(z_all, starts, score_all)
        profiled_e, _ = select_best_profile(profiled_e_all.view(-1, 1, 1), starts, score_all)
        profiled_e = profiled_e.flatten()
        code, _ = select_best_profile(code_all.view(-1, 1, 1), starts, score_all)
        code = code.flatten()
        static_e = static_energy(model, y, schedule, profile_cfg, inference_cfg.amp, inference_cfg.amp_dtype)
        correction = y - z_best
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
