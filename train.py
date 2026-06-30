"""Train TNP-Diffusion on clean normal cycles only."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_PATH,
    DataConfig,
    DiffusionConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
    ensure_dir,
    to_plain_dict,
)
from data import create_dataloader, fit_standard_scaler, infer_measurement_columns, load_cycle_tensors
from engine import ModelEMA, VPSchedule, save_json, set_seed, train_one_epoch
from model import ScoreTransformer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train measurement-only TNP-Diffusion on train_normal.parquet.")
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", type=str, default="tnp_diffusion_v1")

    parser.add_argument("--target-length", type=int, default=384)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--normalize-clip", type=float, default=None)

    parser.add_argument("--model-dim", type=int, default=160)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--heads", type=int, default=5)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)

    parser.add_argument("--beta-min", type=float, default=0.1)
    parser.add_argument("--beta-max", type=float, default=20.0)
    parser.add_argument("--t-eps", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=177)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", type=str, choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=20)
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


def make_optimizer(model: torch.nn.Module, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    try:
        return torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.95),
            fused=train_cfg.device == "cuda" and torch.cuda.is_available(),
        )
    except TypeError:
        return torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            betas=(0.9, 0.95),
        )


def checkpoint_payload(
    model: ScoreTransformer,
    ema: ModelEMA,
    scaler_state: dict[str, Any],
    channels: list[str],
    cfg: ExperimentConfig,
    history: list[dict[str, Any]],
    epoch: int,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict(),
        "scaler": scaler_state,
        "channels": channels,
        "config": cfg.to_dict(),
        "history": history,
    }


def save_checkpoint(
    path: Path,
    model: ScoreTransformer,
    ema: ModelEMA,
    scaler_state: dict[str, Any],
    channels: list[str],
    cfg: ExperimentConfig,
    history: list[dict[str, Any]],
    epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, ema, scaler_state, channels, cfg, history, epoch), path)


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    configure_torch(device)
    set_seed(args.seed)

    run_dir = ensure_dir(Path(args.output_dir) / args.run_name)
    print(f"[info] output_dir={run_dir}")
    print(f"[info] train_path={args.train_path}")

    channels = infer_measurement_columns(args.train_path)
    data_cfg = DataConfig(
        target_length=args.target_length,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        normalize_clip=args.normalize_clip,
        limit_train_samples=args.limit_train_samples,
    )
    model_cfg = ModelConfig(
        input_channels=len(channels),
        target_length=args.target_length,
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        t_eps=args.t_eps,
    )
    diffusion_cfg = DiffusionConfig(beta_min=args.beta_min, beta_max=args.beta_max, t_eps=args.t_eps)
    train_cfg = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        ema_decay=args.ema_decay,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        device=str(device),
        log_interval=args.log_interval,
        save_every=args.save_every,
        output_dir=str(args.output_dir),
        run_name=args.run_name,
    )
    cfg = ExperimentConfig(
        train_path=str(args.train_path),
        data=data_cfg,
        model=model_cfg,
        diffusion=diffusion_cfg,
        train=train_cfg,
    )
    save_json(run_dir / "config.json", cfg.to_dict())

    print("[info] fitting StandardScaler from train_normal only")
    scaler = fit_standard_scaler(
        args.train_path,
        channels,
        eps=data_cfg.scaler_eps,
        clip=data_cfg.normalize_clip,
        limit_samples=data_cfg.limit_train_samples,
    )
    print("[info] loading train cycles")
    tensors = load_cycle_tensors(
        args.train_path,
        channels,
        scaler,
        target_length=data_cfg.target_length,
        limit_samples=data_cfg.limit_train_samples,
    )
    loader = create_dataloader(
        tensors.to_dataset(),
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.persistent_workers,
    )

    model = ScoreTransformer(model_cfg).to(device)
    print(f"[info] model_params={model.parameter_count():,}")
    optimizer = make_optimizer(model, train_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(train_cfg.epochs, 1))
    ema = ModelEMA(model, decay=train_cfg.ema_decay)
    schedule = VPSchedule(diffusion_cfg)

    history: list[dict[str, Any]] = []
    start_time = time.time()
    scaler_state = scaler.to_state_dict()
    for epoch in range(1, train_cfg.epochs + 1):
        metrics = train_one_epoch(model, loader, optimizer, schedule, train_cfg, device, ema, epoch)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "elapsed_sec": float(time.time() - start_time),
            **metrics,
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] loss={row['loss']:.6f} eps_mse={row['eps_mse']:.6f} "
            f"t_mean={row['t_mean']:.4f} lr={row['lr']:.3e}"
        )
        save_json(run_dir / "train_history.json", history)
        if epoch % train_cfg.save_every == 0 or epoch == train_cfg.epochs:
            save_checkpoint(run_dir / f"checkpoint_epoch_{epoch:03d}.pt", model, ema, scaler_state, channels, cfg, history, epoch)
            save_checkpoint(run_dir / "checkpoint_last.pt", model, ema, scaler_state, channels, cfg, history, epoch)

    save_checkpoint(run_dir / "checkpoint_final.pt", model, ema, scaler_state, channels, cfg, history, train_cfg.epochs)
    save_json(run_dir / "train_summary.json", {"run_dir": str(run_dir), "config": to_plain_dict(cfg), "history": history})
    print(f"[done] final_checkpoint={run_dir / 'checkpoint_final.pt'}")


if __name__ == "__main__":
    main()
