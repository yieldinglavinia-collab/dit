"""Inference and evaluation for a trained TNP-Diffusion checkpoint."""

from __future__ import annotations

import argparse
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import (
    DEFAULT_CLEAN_TEST_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VAL_PATH,
    DRIFTED_DIR,
    DataConfig,
    DiffusionConfig,
    InferenceConfig,
    ModelConfig,
    ProfileConfig,
    ensure_dir,
    parse_eval_pairs,
    q099_mixed_eval_paths,
    to_plain_dict,
)
from data import StandardScalerStats, create_dataloader, load_cycle_tensors
from engine import ModelEMA, VPSchedule, metrics_at_tau, save_json, score_dataloader, set_seed
from model import ScoreTransformer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score val/test parquet files with TNP-Diffusion.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--eval", action="append", default=[], help="Evaluation item as NAME=/path/file.parquet. Can repeat.")
    parser.add_argument("--include-q099-drifted", action="store_true", help="Evaluate clean test plus all q=0.99 drifted mixed tests.")
    parser.add_argument("--clean-test-path", type=Path, default=DEFAULT_CLEAN_TEST_PATH)
    parser.add_argument("--drifted-dir", type=Path, default=DRIFTED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "inference_q099")

    parser.add_argument("--threshold-q", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=177)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--use-raw-weights", action="store_true", help="Use non-EMA weights from the checkpoint.")

    parser.add_argument("--profile-mode", choices=("profiled", "static"), default=None)
    parser.add_argument("--score-probes", type=int, default=None)
    parser.add_argument("--score-chunk", type=int, default=None)
    parser.add_argument("--posterior-probes", type=int, default=None)
    parser.add_argument("--posterior-chunk", type=int, default=None)
    parser.add_argument("--nuisance-time-components", type=int, default=None)
    parser.add_argument("--nuisance-noise-probes", type=int, default=None)
    parser.add_argument("--profile-steps", type=int, default=None)
    parser.add_argument("--profile-lr", type=float, default=None)
    parser.add_argument("--profile-starts", type=int, default=None)
    parser.add_argument("--profile-start-scale", type=float, default=None)
    parser.add_argument("--profile-energy-probes", type=int, default=None)
    parser.add_argument("--posterior-t-min", type=float, default=None)
    parser.add_argument("--posterior-t-max", type=float, default=None)
    parser.add_argument("--nuisance-var-floor", type=float, default=None)
    parser.add_argument("--nuisance-var-scale", type=float, default=None)
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA was requested but is not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def configure_torch(device: torch.device) -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def dataclass_from_dict(cls: type, state: dict[str, Any] | None):
    state = state or {}
    names = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in state.items() if key in names})


def override_profile(profile: ProfileConfig, args: argparse.Namespace) -> ProfileConfig:
    overrides = {
        "profile_score_mode": args.profile_mode,
        "score_probes": args.score_probes,
        "score_chunk": args.score_chunk,
        "posterior_probes": args.posterior_probes,
        "posterior_chunk": args.posterior_chunk,
        "nuisance_time_components": args.nuisance_time_components,
        "nuisance_noise_probes": args.nuisance_noise_probes,
        "profile_steps": args.profile_steps,
        "profile_lr": args.profile_lr,
        "profile_starts": args.profile_starts,
        "profile_start_scale": args.profile_start_scale,
        "profile_energy_probes": args.profile_energy_probes,
        "posterior_t_min": args.posterior_t_min,
        "posterior_t_max": args.posterior_t_max,
        "nuisance_var_floor": args.nuisance_var_floor,
        "nuisance_var_scale": args.nuisance_var_scale,
    }
    state = to_plain_dict(profile)
    for key, value in overrides.items():
        if value is not None:
            state[key] = value
    return ProfileConfig(**state)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def check_paths(paths: dict[str, Path]) -> None:
    missing = [f"{name}={path}" for name, path in paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing evaluation parquet(s): " + "; ".join(missing))


def make_loader(
    path: Path,
    channels: list[str],
    scaler: StandardScalerStats,
    data_cfg: DataConfig,
    infer_cfg: InferenceConfig,
    device: torch.device,
) -> tuple[pd.DataFrame | None, Any]:
    tensors = load_cycle_tensors(
        path,
        channels,
        scaler,
        target_length=data_cfg.target_length,
        limit_samples=data_cfg.limit_eval_samples,
    )
    loader = create_dataloader(
        tensors.to_dataset(),
        batch_size=infer_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=data_cfg.num_workers > 0,
    )
    return None, loader


def score_path(
    name: str,
    path: Path,
    model: ScoreTransformer,
    schedule: VPSchedule,
    scaler: StandardScalerStats,
    channels: list[str],
    data_cfg: DataConfig,
    infer_cfg: InferenceConfig,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    print(f"[score] {name}: {path}")
    _, loader = make_loader(path, channels, scaler, data_cfg, infer_cfg, device)
    scores = score_dataloader(model, loader, schedule, infer_cfg, device, dataset_name=name)
    scores.to_csv(output_dir / f"{safe_name(name)}_scores.csv", index=False)
    return scores


def threshold_from_val(val_scores: pd.DataFrame, q: float) -> dict[str, Any]:
    scores = val_scores["score"].to_numpy(dtype=np.float64)
    tau = float(np.quantile(scores, q))
    alarms = scores > tau
    return {
        "threshold_q": float(q),
        "tau": tau,
        "calibration_count": int(scores.shape[0]),
        "calibration_far_strict_gt": float(alarms.mean()),
        "score_min": float(np.min(scores)),
        "score_median": float(np.median(scores)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "score_max": float(np.max(scores)),
    }


def add_clean_reference(metrics_rows: list[dict[str, Any]], clean_name: str = "clean_test") -> list[dict[str, Any]]:
    clean = next((row for row in metrics_rows if row["split"] == clean_name), None)
    if clean is None:
        return metrics_rows
    enriched = []
    for row in metrics_rows:
        next_row = dict(row)
        for key in ("FAR", "TPR", "AUROC", "AP", "score_mean", "score_median"):
            if key in row and key in clean:
                next_row[f"delta_vs_clean_{key}"] = float(row[key] - clean[key])
        enriched.append(next_row)
    return enriched


def plot_metric_bars(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    for metric in ("FAR", "TPR", "AUROC", "AP"):
        if metric not in metrics_df.columns:
            continue
        present = metrics_df.dropna(subset=[metric])
        if present.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(10, 0.42 * len(present)), 4.6))
        ax.bar(present["split"], present[metric], color="#4477AA")
        ax.set_ylabel(metric)
        ax.set_ylim(0.0, min(1.0, max(0.05, float(present[metric].max()) * 1.12)))
        ax.tick_params(axis="x", rotation=60)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric.lower()}_bar.png", dpi=180)
        plt.close(fig)


def plot_score_distributions(score_tables: dict[str, pd.DataFrame], tau: float, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, scores in score_tables.items():
        if name != "val_normal" and name != "clean_test" and not name.endswith("_severe"):
            continue
        ax.hist(scores["score"], bins=40, histtype="step", density=True, linewidth=1.4, label=name)
    ax.axvline(tau, color="black", linestyle="--", linewidth=1.2, label="tau")
    ax.set_xlabel("TNP score")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "score_distribution_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    configure_torch(device)
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)

    checkpoint = load_checkpoint(args.checkpoint)
    cfg_state = checkpoint.get("config", {})
    data_cfg = dataclass_from_dict(DataConfig, cfg_state.get("data"))
    model_cfg = dataclass_from_dict(ModelConfig, cfg_state.get("model"))
    diffusion_cfg = dataclass_from_dict(DiffusionConfig, cfg_state.get("diffusion"))
    data_cfg.num_workers = args.num_workers
    data_cfg.pin_memory = device.type == "cuda"
    data_cfg.persistent_workers = args.num_workers > 0
    data_cfg.limit_eval_samples = args.limit_eval_samples

    profile_cfg = dataclass_from_dict(ProfileConfig, cfg_state.get("inference", {}).get("profile") if cfg_state.get("inference") else None)
    profile_cfg = override_profile(profile_cfg, args)
    infer_cfg = InferenceConfig(
        seed=args.seed,
        threshold_q=args.threshold_q,
        batch_size=args.batch_size,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        device=str(device),
        output_dir=str(output_dir),
        profile=profile_cfg,
    )

    channels = list(checkpoint["channels"])
    scaler = StandardScalerStats.from_state_dict(checkpoint["scaler"])
    model_cfg.input_channels = len(channels)
    model = ScoreTransformer(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if not args.use_raw_weights and "ema_state" in checkpoint:
        ModelEMA.apply_to(model, checkpoint["ema_state"])
        weight_source = "ema_state"
    else:
        weight_source = "model_state"
    model.eval()
    schedule = VPSchedule(diffusion_cfg)

    run_config = {
        "checkpoint": str(args.checkpoint),
        "weight_source": weight_source,
        "calibration": str(args.calibration),
        "output_dir": str(output_dir),
        "data": to_plain_dict(data_cfg),
        "model": to_plain_dict(model_cfg),
        "diffusion": to_plain_dict(diffusion_cfg),
        "inference": to_plain_dict(infer_cfg),
    }
    save_json(output_dir / "run_config.json", run_config)

    check_paths({"val_normal": args.calibration})
    val_scores = score_path(
        "val_normal",
        args.calibration,
        model,
        schedule,
        scaler,
        channels,
        data_cfg,
        infer_cfg,
        device,
        output_dir,
    )
    threshold = threshold_from_val(val_scores, args.threshold_q)
    tau = float(threshold["tau"])
    save_json(output_dir / "threshold.json", threshold)
    print(f"[threshold] q={args.threshold_q:.4f} tau={tau:.6f} val_FAR={threshold['calibration_far_strict_gt']:.6f}")

    eval_paths = parse_eval_pairs(args.eval)
    if args.include_q099_drifted:
        eval_paths.update(q099_mixed_eval_paths(args.clean_test_path, args.drifted_dir))
    if not eval_paths:
        eval_paths["clean_test"] = args.clean_test_path
    check_paths(eval_paths)

    score_tables: dict[str, pd.DataFrame] = {"val_normal": val_scores}
    metrics_rows: list[dict[str, Any]] = []
    for name, path in eval_paths.items():
        scores = score_path(name, path, model, schedule, scaler, channels, data_cfg, infer_cfg, device, output_dir)
        score_tables[name] = scores
        row = {"split": name, "path": str(path), **metrics_at_tau(scores, tau)}
        metrics_rows.append(row)
        print(
            f"[metrics] {name} FAR={row.get('FAR', float('nan')):.4f} "
            f"TPR={row.get('TPR', float('nan')):.4f} AUROC={row.get('AUROC', float('nan')):.4f}"
        )

    metrics_rows = add_clean_reference(metrics_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
    save_json(output_dir / "metrics_summary.json", metrics_rows)
    plot_metric_bars(metrics_df, output_dir)
    plot_score_distributions(score_tables, tau, output_dir)
    print(f"[done] metrics={output_dir / 'metrics_summary.csv'}")


if __name__ == "__main__":
    main()
