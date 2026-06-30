#!/usr/bin/env python3
"""Run CCSD-QKV inference and q=0.99 threshold evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch

from config import (
    DEFAULT_CLEAN_TEST_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VAL_PATH,
    DataConfig,
    DiffusionConfig,
    InferenceConfig,
    ModelConfig,
    ensure_dir,
    parse_eval_pairs,
    q099_eval_paths,
)
from data import StandardScalerStats, create_dataloader, load_cycle_tensors
from engine import metrics_at_threshold, resolve_device, save_json, score_dataloader, set_seed
from model import CCSDQKVDenoiser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_VAL_PATH)
    parser.add_argument("--eval", nargs="*", default=[], help="Optional NAME=/path/to/file.parquet eval pairs.")
    parser.add_argument("--include-q099-drifted", action="store_true", help="Evaluate clean_test plus all q=0.99 drifted mixed tests.")
    parser.add_argument("--clean-test-path", type=Path, default=DEFAULT_CLEAN_TEST_PATH)
    parser.add_argument("--drifted-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold-q", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mc-samples", type=int, default=None)
    parser.add_argument("--mc-chunk", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default=None)
    parser.add_argument("--use-raw-weights", action="store_true", help="Use raw model weights instead of EMA weights.")
    parser.add_argument("--limit-eval-samples", type=int, default=None)
    return parser


def restore_model_config(payload: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**payload)


def restore_diffusion_config(payload: dict[str, Any]) -> DiffusionConfig:
    return DiffusionConfig(**payload)


def restore_data_config(payload: dict[str, Any]) -> DataConfig:
    return DataConfig(**payload)


def restore_inference_config(payload: dict[str, Any]) -> InferenceConfig:
    return InferenceConfig(**payload)


def merge_inference_args(saved: InferenceConfig, args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        seed=saved.seed,
        batch_size=args.batch_size or saved.batch_size,
        mc_samples=args.mc_samples or saved.mc_samples,
        mc_chunk=args.mc_chunk or saved.mc_chunk,
        threshold_q=args.threshold_q,
        amp=saved.amp if args.amp is None else bool(args.amp),
        amp_dtype=args.amp_dtype or saved.amp_dtype,
        device=args.device or saved.device,
        output_dir=str(args.output_dir or saved.output_dir),
    )


def main() -> None:
    args = build_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_cfg = restore_model_config(config["model"])
    diffusion_cfg = restore_diffusion_config(config["diffusion"])
    data_cfg = restore_data_config(config["data"])
    saved_inference = restore_inference_config(config["inference"])
    inference_cfg = merge_inference_args(saved_inference, args)

    device = resolve_device(inference_cfg.device)
    set_seed(inference_cfg.seed)
    channels = list(checkpoint["channels"])
    scaler = StandardScalerStats.from_state_dict(checkpoint["scaler"])

    model = CCSDQKVDenoiser(model_cfg).to(device)
    if args.use_raw_weights or checkpoint.get("ema_state") is None:
        model.load_state_dict(checkpoint["model_state"])
        weight_source = "raw"
    else:
        state = checkpoint["model_state"]
        state.update({key: value for key, value in checkpoint["ema_state"].items() if key in state})
        model.load_state_dict(state)
        weight_source = "ema"
    model.eval()

    output_dir = ensure_dir(args.output_dir or (args.checkpoint.parent / "inference_q099"))
    print(f"Using {weight_source} weights.")
    print(f"Scoring calibration: {args.calibration}")
    val_scores = score_path(
        name="val_normal",
        path=args.calibration,
        model=model,
        channels=channels,
        scaler=scaler,
        data_cfg=data_cfg,
        diffusion_cfg=diffusion_cfg,
        inference_cfg=inference_cfg,
        device=device,
        num_workers=args.num_workers,
        limit_samples=args.limit_eval_samples,
    )
    tau = float(val_scores["score"].quantile(inference_cfg.threshold_q))
    val_scores.to_csv(output_dir / "val_normal_scores.csv", index=False)
    save_json(
        output_dir / "threshold.json",
        {
            "threshold_q": inference_cfg.threshold_q,
            "tau": tau,
            "val_clean_FAR": float((val_scores["score"] > tau).mean()),
            "calibration_path": str(args.calibration),
        },
    )

    eval_paths = parse_eval_pairs(args.eval) if args.eval else {}
    if args.include_q099_drifted:
        drifted_dir = args.drifted_dir if args.drifted_dir is not None else None
        eval_paths.update(q099_eval_paths(clean_test_path=args.clean_test_path, drifted_dir=drifted_dir) if drifted_dir else q099_eval_paths(args.clean_test_path))
    if not eval_paths:
        eval_paths["clean_test"] = args.clean_test_path

    summary_rows = []
    for name, path in eval_paths.items():
        print(f"Scoring {name}: {path}")
        scores = score_path(
            name=name,
            path=path,
            model=model,
            channels=channels,
            scaler=scaler,
            data_cfg=data_cfg,
            diffusion_cfg=diffusion_cfg,
            inference_cfg=inference_cfg,
            device=device,
            num_workers=args.num_workers,
            limit_samples=args.limit_eval_samples,
        )
        scores_path = output_dir / f"{name}_scores.csv"
        scores.to_csv(scores_path, index=False)
        metrics_dict = metrics_at_threshold(scores, tau)
        metrics_dict.update({"dataset": name, "path": str(path), "scores_path": str(scores_path)})
        summary_rows.append(metrics_dict)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    save_json(output_dir / "metrics_summary.json", summary.to_dict(orient="records"))
    save_json(
        output_dir / "run_config.json",
        {
            "checkpoint": str(args.checkpoint),
            "weight_source": weight_source,
            "inference": inference_cfg.__dict__,
            "eval_paths": {name: str(path) for name, path in eval_paths.items()},
        },
    )
    plot_summary(summary, output_dir)
    print(summary.to_string(index=False))
    print(f"Saved inference outputs to {output_dir}")


def score_path(
    name: str,
    path: Path,
    model: CCSDQKVDenoiser,
    channels: list[str],
    scaler: StandardScalerStats,
    data_cfg: DataConfig,
    diffusion_cfg: DiffusionConfig,
    inference_cfg: InferenceConfig,
    device: torch.device,
    num_workers: int,
    limit_samples: int | None,
) -> pd.DataFrame:
    tensors = load_cycle_tensors(
        parquet_path=path,
        channels=channels,
        scaler=scaler,
        target_length=data_cfg.target_length,
        limit_samples=limit_samples,
    )
    loader = create_dataloader(
        tensors.to_dataset(),
        batch_size=inference_cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.persistent_workers,
    )
    return score_dataloader(model, loader, diffusion_cfg, inference_cfg, device, name)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    for metric, ylabel, filename in [
        ("FAR", "FAR", "far_by_dataset.png"),
        ("TPR", "TPR", "tpr_by_dataset.png"),
        ("AUROC", "AUROC", "auroc_by_dataset.png"),
        ("AP", "AP", "ap_by_dataset.png"),
    ]:
        if metric not in summary.columns or summary[metric].dropna().empty:
            continue
        data = summary.dropna(subset=[metric]).copy()
        plt.figure(figsize=(12, 5.2))
        plt.bar(data["dataset"], data[metric], color="#4C78A8")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"{metric} by Dataset")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()


if __name__ == "__main__":
    main()
