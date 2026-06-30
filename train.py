#!/usr/bin/env python3
"""Train CCSD-QKV on clean train_normal only."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from config import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_PATH,
    DataConfig,
    DiffusionConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    TrainConfig,
    ensure_dir,
)
from data import (
    create_dataloader,
    fit_standard_scaler_from_parquet,
    infer_measurement_columns,
    load_cycle_tensors,
)
from engine import ModelEMA, resolve_device, save_json, set_seed, train_one_epoch
from model import CCSDQKVDenoiser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default="ccsd_qkv_v1")
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--train-mask-ratio", type=float, default=0.25)
    parser.add_argument("--eval-mask-ratio", type=float, default=0.25)
    parser.add_argument("--sigma-min", type=float, default=0.02)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--mask-strategy", choices=("element", "time_block", "patch", "mixed"), default="mixed")
    parser.add_argument("--seed", type=int, default=177)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = resolve_device(args.device)
    set_seed(args.seed)

    channels = infer_measurement_columns(args.train_path)
    if len(channels) != 58:
        raise RuntimeError(f"Expected 58 measurement channels, found {len(channels)}.")

    data_cfg = DataConfig(
        target_length=args.target_length,
        num_workers=args.num_workers,
        limit_train_samples=args.limit_train_samples,
    )
    model_cfg = ModelConfig(
        input_channels=len(channels),
        target_length=args.target_length,
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        sigma_embed_dim=args.model_dim,
    )
    diffusion_cfg = DiffusionConfig(
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        train_mask_ratio=args.train_mask_ratio,
        eval_mask_ratio=args.eval_mask_ratio,
        mask_strategy=args.mask_strategy,
    )
    train_cfg = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        device=str(device),
        ema_decay=args.ema_decay,
        save_every=args.save_every,
        output_dir=str(args.output_dir),
        run_name=args.run_name,
    )
    inference_cfg = replace(InferenceConfig(), device=str(device), amp=args.amp, amp_dtype=args.amp_dtype)
    exp_cfg = ExperimentConfig(
        train_path=str(args.train_path),
        data=data_cfg,
        model=model_cfg,
        diffusion=diffusion_cfg,
        train=train_cfg,
        inference=inference_cfg,
    )

    print("Fitting train_normal StandardScaler...")
    scaler = fit_standard_scaler_from_parquet(
        args.train_path,
        channels=channels,
        eps=data_cfg.scaler_eps,
        clip=data_cfg.normalize_clip,
        limit_samples=args.limit_train_samples,
    )
    print("Loading fixed-length train cycles...")
    train_tensors = load_cycle_tensors(
        args.train_path,
        channels=channels,
        scaler=scaler,
        target_length=args.target_length,
        limit_samples=args.limit_train_samples,
    )
    loader = create_dataloader(
        train_tensors.to_dataset(),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=data_cfg.persistent_workers,
    )

    model = CCSDQKVDenoiser(model_cfg).to(device)
    print(f"Model parameters: {model.parameter_count():,}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=device.type == "cuda")
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0.0 else None

    output_dir = ensure_dir(args.output_dir / args.run_name)
    save_json(output_dir / "config.json", exp_cfg.to_dict())
    history: list[dict[str, float]] = []

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs")
    for epoch in epoch_bar:
        metrics = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            diffusion_cfg=diffusion_cfg,
            train_cfg=train_cfg,
            device=device,
            ema=ema,
            epoch=epoch,
        )
        scheduler.step()
        metrics["epoch"] = epoch
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(metrics)
        epoch_bar.set_postfix(loss=f"{metrics['loss']:.4f}", lr=f"{metrics['lr']:.2e}")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"checkpoint_epoch_{epoch:04d}.pt", model, ema, scaler, channels, exp_cfg, history)

    checkpoint_path = output_dir / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, ema, scaler, channels, exp_cfg, history)
    save_json(output_dir / "train_history.json", {"history": history})
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved history: {output_dir / 'train_history.json'}")


def save_checkpoint(
    path: Path,
    model: CCSDQKVDenoiser,
    ema: ModelEMA | None,
    scaler,
    channels: list[str],
    exp_cfg: ExperimentConfig,
    history: list[dict[str, float]],
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "ema_state": ema.state_dict() if ema is not None else None,
        "scaler": scaler.to_state_dict(),
        "channels": channels,
        "config": exp_cfg.to_dict(),
        "history": history,
    }
    torch.save(checkpoint, path)


if __name__ == "__main__":
    main()
