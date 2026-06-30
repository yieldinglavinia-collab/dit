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


def sample_train_time(batch: int, cfg: DiffusionConfig, device: torch.device) -> torch.Tensor:
    """Diffusion time law pi(t), shared by DSM training and nuisance mixture."""
    return torch.empty(batch, device=device).uniform_(cfg.t_eps, 1.0)


def diffusion_time_grid(cfg: DiffusionConfig, steps: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic quadrature grid and weights for the uniform diffusion time law pi(t)."""
    t_values = torch.linspace(cfg.t_eps, 1.0, steps=max(int(steps), 2), device=device, dtype=dtype)
    weights = torch.full((t_values.numel(),), 1.0 / t_values.numel(), device=device, dtype=dtype)
    return t_values, weights


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


def denoising_mean(model: ScoreTransformer, u: torch.Tensor, t: torch.Tensor, schedule: VPSchedule) -> torch.Tensor:
    """Tweedie denoising mean D_theta(u,t) = (u + sigma_t^2 s_theta(u,t)) / alpha_t."""
    alpha, sigma = schedule.alpha_sigma(t)
    score = model.score(u, t, schedule)
    return (u + sigma.pow(2).view(-1, 1, 1) * score) / alpha.view(-1, 1, 1).clamp_min(1e-5)


def probability_flow_drift(model: ScoreTransformer, x_t: torch.Tensor, t: torch.Tensor, schedule: VPSchedule) -> torch.Tensor:
    """VP probability-flow ODE drift f(x,t) - 0.5 g(t)^2 s_theta(x,t)."""
    beta = schedule.beta(t).view(-1, 1, 1)
    score = model.score(x_t, t, schedule)
    return -0.5 * beta * x_t - 0.5 * beta * score


def _hutchinson_divergence(drift: torch.Tensor, x_t: torch.Tensor, probe: torch.Tensor, create_graph: bool) -> torch.Tensor:
    """Per-sample Hutchinson trace estimate for div_x drift(x,t)."""
    vjp = torch.autograd.grad(
        outputs=(drift * probe).sum(),
        inputs=x_t,
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return (vjp * probe).flatten(1).sum(dim=1)


def probability_flow_nll_fixed(
    model: ScoreTransformer,
    x0: torch.Tensor,
    schedule: VPSchedule,
    t_values: torch.Tensor,
    probes: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """Probability-flow ODE total negative log likelihood E_theta(x)."""
    if t_values.numel() < 2:
        raise ValueError("probability-flow likelihood requires at least two time values")
    was_training = model.training
    model.eval()
    x = x0
    integral_div = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
    for idx in range(t_values.numel() - 1):
        t0 = t_values[idx]
        t1 = t_values[idx + 1]
        dt = t1 - t0
        x = x.requires_grad_(True)
        t = torch.full((x.shape[0],), float(t0.item()), device=x.device, dtype=x.dtype)
        drift = probability_flow_drift(model, x, t, schedule)
        div = _hutchinson_divergence(drift, x, probes[idx], create_graph=create_graph)
        integral_div = integral_div + div * dt
        x = x + drift * dt
        if not create_graph:
            x = x.detach()
    dim = x0[0].numel()
    log_p1 = -0.5 * (x.flatten(1).pow(2).sum(dim=1) + dim * math.log(2.0 * math.pi))
    if was_training:
        model.train()
    return -(log_p1 + integral_div)


def probability_flow_nll(
    model: ScoreTransformer,
    values: torch.Tensor,
    schedule: VPSchedule,
    steps: int,
    hutchinson_probes: int,
    hutchinson_chunk: int,
    create_graph: bool = False,
) -> torch.Tensor:
    """Average probability-flow total NLL across Hutchinson probes, evaluated in probe chunks."""
    t_values = torch.linspace(schedule.cfg.t_eps, 1.0, steps=max(int(steps), 2), device=values.device, dtype=values.dtype)
    estimates = []
    total_probes = max(int(hutchinson_probes), 1)
    chunk = max(int(hutchinson_chunk), 1)
    for start in range(0, total_probes, chunk):
        count = min(chunk, total_probes - start)
        for _ in range(count):
            probes = torch.randn(t_values.numel() - 1, *values.shape, device=values.device, dtype=values.dtype)
            estimates.append(probability_flow_nll_fixed(model, values, schedule, t_values, probes, create_graph=create_graph))
    return torch.stack(estimates, dim=0).mean(dim=0)


def clean_energy(
    model: ScoreTransformer,
    values: torch.Tensor,
    schedule: VPSchedule,
    inference_cfg: InferenceConfig,
    probes: int,
    create_graph: bool,
) -> torch.Tensor:
    return probability_flow_nll(
        model,
        values,
        schedule,
        steps=schedule.cfg.likelihood_steps,
        hutchinson_probes=probes,
        hutchinson_chunk=inference_cfg.profile.score_chunk,
        create_graph=create_graph,
    )


def posterior_covariance(
    model: ScoreTransformer,
    u: torch.Tensor,
    t: torch.Tensor,
    schedule: VPSchedule,
    create_graph: bool,
) -> torch.Tensor:
    """Full denoising posterior covariance Sigma_theta(u,t).

    Uses the Tweedie/Jacobian identity Cov(x0|u) = (sigma_t^2 / alpha_t) dD_theta(u,t)/du.
    The result is symmetrized and receives a tiny numerical jitter for Cholesky stability.
    """
    covariances = []
    flat_dim = u[0].numel()
    for index in range(u.shape[0]):
        ui = u[index : index + 1]
        ti = t[index : index + 1]

        def flat_denoiser(flat_u: torch.Tensor) -> torch.Tensor:
            shaped = flat_u.view_as(ui)
            return denoising_mean(model, shaped, ti, schedule).flatten()

        jac = torch.autograd.functional.jacobian(flat_denoiser, ui.flatten(), create_graph=create_graph, vectorize=True)
        _, sigma = schedule.alpha_sigma(ti)
        alpha, _ = schedule.alpha_sigma(ti)
        cov = (sigma.pow(2) / alpha.clamp_min(1e-5)).view(1, 1) * jac
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(cov)
        cov = (eigvecs * eigvals.clamp_min(1e-5).unsqueeze(0)) @ eigvecs.transpose(-1, -2)
        jitter = torch.eye(flat_dim, device=u.device, dtype=u.dtype) * 1e-6
        covariances.append(cov + jitter)
    return torch.stack(covariances, dim=0)


def nuisance_component_nll(delta_flat: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    """Full multivariate Gaussian -log N(delta; 0, K)."""
    dim = delta_flat.shape[1]
    chol, info = torch.linalg.cholesky_ex(covariance)
    if bool((info > 0).any()):
        jitter = torch.eye(dim, device=delta_flat.device, dtype=delta_flat.dtype).unsqueeze(0) * 1e-3
        chol = torch.linalg.cholesky(covariance + jitter)
    solution = torch.cholesky_solve(delta_flat.unsqueeze(-1), chol).squeeze(-1)
    quadratic = (delta_flat * solution).sum(dim=1)
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(dim=1)
    return 0.5 * (quadratic + logdet + dim * math.log(2.0 * math.pi))


def nuisance_code(
    model: ScoreTransformer,
    delta: torch.Tensor,
    z: torch.Tensor,
    schedule: VPSchedule,
    inference_cfg: InferenceConfig,
    create_graph: bool,
) -> torch.Tensor:
    """C_theta(delta|z) = -log int N(delta;0,K_theta(z,t)) pi(t) dt."""
    t_grid, weights = diffusion_time_grid(schedule.cfg, max(2, inference_cfg.profile.nuisance_time_steps), z.device, z.dtype)
    delta_flat = delta.flatten(1)
    log_components = []
    for t_scalar, weight in zip(t_grid, weights):
        t = torch.full((z.shape[0],), float(t_scalar.item()), device=z.device, dtype=z.dtype)
        alpha, _ = schedule.alpha_sigma(t)
        u = alpha.view(-1, 1, 1) * z
        covariance = posterior_covariance(model, u, t, schedule, create_graph=create_graph)
        component_nll = nuisance_component_nll(delta_flat, covariance)
        log_components.append(torch.log(weight) - component_nll)
    return -torch.logsumexp(torch.stack(log_components, dim=0), dim=0)


def tnp_objective(
    model: ScoreTransformer,
    y: torch.Tensor,
    z: torch.Tensor,
    schedule: VPSchedule,
    inference_cfg: InferenceConfig,
    energy_probes: int,
    create_graph: bool,
) -> dict[str, torch.Tensor]:
    energy = clean_energy(model, z, schedule, inference_cfg, probes=energy_probes, create_graph=create_graph)
    code = nuisance_code(model, y - z, z, schedule, inference_cfg, create_graph=create_graph)
    score = energy + code
    return {"score": score, "profiled_energy": energy, "nuisance_code": code}


@torch.no_grad()
def static_energy(
    model: ScoreTransformer,
    values: torch.Tensor,
    schedule: VPSchedule,
    profile_cfg: ProfileConfig,
    amp: bool,
    amp_dtype: str,
) -> torch.Tensor:
    """Clean-normal energy E_theta(x) from probability-flow ODE NLL."""
    del amp, amp_dtype
    with torch.enable_grad():
        return probability_flow_nll(
            model,
            values.detach(),
            schedule,
            steps=schedule.cfg.likelihood_steps,
            hutchinson_probes=profile_cfg.score_probes,
            hutchinson_chunk=profile_cfg.score_chunk,
            create_graph=False,
        ).detach()


def profile_score_batch(
    model: ScoreTransformer,
    y: torch.Tensor,
    schedule: VPSchedule,
    inference_cfg: InferenceConfig,
) -> dict[str, torch.Tensor]:
    """Computes S_TNP(y), z*, and delta* for one batch."""
    profile_cfg = inference_cfg.profile
    z = y.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=profile_cfg.profile_lr)

    for _ in range(profile_cfg.profile_steps):
        optimizer.zero_grad(set_to_none=True)
        result = tnp_objective(
            model,
            y,
            z,
            schedule,
            inference_cfg,
            energy_probes=profile_cfg.profile_energy_probes,
            create_graph=True,
        )
        result["score"].mean().backward()
        optimizer.step()

    final = tnp_objective(
        model,
        y,
        z,
        schedule,
        inference_cfg,
        energy_probes=profile_cfg.score_probes,
        create_graph=False,
    )
    static_e = static_energy(model, y, schedule, profile_cfg, inference_cfg.amp, inference_cfg.amp_dtype)
    correction = y - z
    correction_rms = correction.pow(2).flatten(1).mean(dim=1).sqrt()
    return {
        "score": final["score"].detach(),
        "static_energy": static_e.detach(),
        "profiled_energy": final["profiled_energy"].detach(),
        "nuisance_code": final["nuisance_code"].detach(),
        "correction_rms": correction_rms.detach(),
        "profile_improvement": (static_e - final["score"]).detach(),
        "z_star": z.detach(),
        "delta_star": correction.detach(),
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
    z_star_chunks: list[torch.Tensor] = []
    delta_star_chunks: list[torch.Tensor] = []
    progress = tqdm(loader, desc=f"score:{dataset_name}", leave=False)
    for batch in progress:
        batch = move_to_device(batch, device)
        result = profile_score_batch(model, batch["values"], schedule, inference_cfg)
        z_star_chunks.append(result["z_star"].detach().cpu())
        delta_star_chunks.append(result["delta_star"].detach().cpu())
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
    frame = pd.DataFrame(rows).sort_values("sample").reset_index(drop=True)
    if z_star_chunks:
        frame.attrs["z_star"] = torch.cat(z_star_chunks, dim=0).numpy()
        frame.attrs["delta_star"] = torch.cat(delta_star_chunks, dim=0).numpy()
    return frame


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
