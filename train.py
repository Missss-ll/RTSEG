

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config, load_config
from data.semantickitti import build_dataloaders
from models import RMTSeg, RMTLoss, rmt_seg_tiny, rmt_seg_small, rmt_seg_base
from utils.reliability import compute_reliability
from utils.metrics import SegmentationMetrics
from utils.checkpoint import save_checkpoint, load_checkpoint

VARIANT_MAP = {
    "rmt_seg_tiny": rmt_seg_tiny,
    "rmt_seg_small": rmt_seg_small,
    "rmt_seg_base": rmt_seg_base,
}

def build_model(cfg: Config) -> RMTSeg:
    kwargs = cfg.model.to_dict()

    variant = cfg.model.variant
    if variant in VARIANT_MAP:
        factory = VARIANT_MAP[variant]
        model = factory()
        for k, v in kwargs.items():
            if hasattr(model, k):
                continue
        return VARIANT_MAP[variant](**kwargs)
    else:
        return RMTSeg(**kwargs)

class EMAModel:

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    @torch.no_grad()
    def assign_to(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]

def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: Config,
    steps_per_epoch: int,
) -> optim.lr_scheduler._LRScheduler:
    sc = cfg.train
    total_steps = sc.epochs * steps_per_epoch
    warmup_steps = sc.warmup_epochs * steps_per_epoch

    if sc.scheduler == "cosine":
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps, eta_min=sc.min_lr
        )
    elif sc.scheduler == "step":
        base_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sc.step_size * steps_per_epoch,
            gamma=sc.step_gamma,
        )
    elif sc.scheduler == "plateau":
        base_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=sc.plateau_patience, factor=0.5
        )
    else:
        raise ValueError(f"Unknown scheduler: {sc.scheduler}")

    if warmup_steps <= 0:
        return base_scheduler

    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, base_scheduler],
        milestones=[warmup_steps],
    )

def train_epoch(
    model: RMTSeg,
    loader: torch.utils.data.DataLoader,
    criterion: RMTLoss,
    optimizer: optim.Optimizer,
    scaler: GradScaler | None,
    cfg: Config,
    ema: EMAModel | None,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_steps = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [train]", leave=False)
    for step, batch in enumerate(pbar):
        rv = batch["range_view"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.no_grad():
            R = compute_reliability(
                rv,
                alpha=cfg.reliability.alpha,
                tau=cfg.reliability.tau,
                beta=cfg.reliability.beta,
                smooth_sigma=cfg.reliability.smooth_sigma,
            )
            D = (1.0 - R) if cfg.model.adaptive_window else None

        if cfg.loss.lambda_ref > 0 and ema is not None:
            with torch.no_grad():
                ref_logits = model(rv)
                ref_logits = ref_logits.detach()
        else:
            ref_logits = torch.zeros_like(
                rv[:, :1, :, :]
            ).expand(-1, cfg.model.num_classes, -1, -1)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                logits = model(rv, D)
                loss, loss_dict = criterion(logits, labels, R, ref_logits)
            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(rv, D)
            loss, loss_dict = criterion(logits, labels, R, ref_logits)
            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.train.grad_clip
                )
            optimizer.step()

        total_loss += loss.item()

        if ema is not None:
            ema.update(model)

        if step % cfg.train.log_interval == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                rel=f"{loss_dict['rel'].item():.4f}",
                ref=f"{loss_dict['ref'].item():.4f}",
            )

    return total_loss / max(total_steps, 1)

@torch.no_grad()
def validate(
    model: RMTSeg,
    loader: torch.utils.data.DataLoader,
    metrics: SegmentationMetrics,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics.reset()

    pbar = tqdm(loader, desc="    [valid]", leave=False)
    for batch in pbar:
        rv = batch["range_view"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        logits = model(rv)
        preds = logits.argmax(dim=1)
        metrics.update(preds, labels)

    return metrics.compute()

def main() -> None:
    parser = argparse.ArgumentParser(description="RTSEG Training")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (overrides config value).")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override dataset path.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (e.g. 'cuda:0', 'cpu').")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.resume:
        cfg.checkpoint.resume = args.resume

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Device: {device}")

    print(f"Loading SemanticKITTI from: {cfg.data.data_root}")
    train_loader, val_loader = build_dataloaders(
        data_root=cfg.data.data_root,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        h=cfg.data.h,
        w=cfg.data.w,
        fov_up=cfg.data.fov_up,
        fov_down=cfg.data.fov_down,
        augment=cfg.data.augment,
    )
    print(f"  Train: {len(train_loader.dataset)} samples, "
          f"{len(train_loader)} batches")
    print(f"  Valid: {len(val_loader.dataset)} samples, "
          f"{len(val_loader)} batches")

    print(f"Building model variant: {cfg.model.variant}")
    model = build_model(cfg)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params / 1e6:.2f} M")

    class_weight = None
    if cfg.loss.class_weight is not None:
        class_weight = torch.load(cfg.loss.class_weight)
        if isinstance(class_weight, torch.Tensor):
            class_weight = class_weight.to(device)

    criterion = RMTLoss(
        num_classes=cfg.model.num_classes,
        lambda_rel=cfg.loss.lambda_rel,
        lambda_ref=cfg.loss.lambda_ref,
        ref_mode=cfg.loss.ref_mode,
        label_smoothing=cfg.loss.label_smoothing,
        weight=class_weight,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.train.betas),
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    scaler = GradScaler("cuda") if cfg.train.amp and device.type == "cuda" else None

    ema = EMAModel(model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None

    start_epoch = 0
    best_mIoU = 0.0
    if cfg.checkpoint.resume:
        print(f"Resuming from: {cfg.checkpoint.resume}")
        info = load_checkpoint(
            model, cfg.checkpoint.resume,
            optimizer=optimizer, scheduler=scheduler,
            map_location=device_str,
        )
        start_epoch = info["epoch"] + 1
        best_mIoU = info["best_metric"]
        print(f"  Resumed at epoch {start_epoch}, best mIoU={best_mIoU:.4f}")

    val_metrics = SegmentationMetrics(
        num_classes=cfg.model.num_classes, ignore_index=255
    )

    save_dir = Path(cfg.checkpoint.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training: {cfg.train.epochs} epochs "
          f"(lr={cfg.train.lr}, bs={cfg.data.batch_size})\n")

    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()

        avg_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, cfg, ema,
            device, epoch,
        )

        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                pass
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        val_results = None
        if (epoch + 1) % cfg.train.val_interval == 0:
            if ema is not None:
                ema_backup = copy.deepcopy(
                    {n: p.data.clone() for n, p in model.named_parameters()}
                )
                ema.assign_to(model)

            val_results = validate(model, val_loader, val_metrics, device)

            if ema is not None:
                for n, p in model.named_parameters():
                    p.data.copy_(ema_backup[n])

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_results["mIoU"])

        elapsed = time.time() - t0
        status = (
            f"Epoch {epoch + 1:3d}/{cfg.train.epochs} | "
            f"loss={avg_loss:.4f} | lr={lr:.2e} | time={elapsed:.1f}s"
        )
        if val_results:
            status += (
                f" | mIoU={val_results['mIoU'] * 100:.2f}%"
                f" | pix_acc={val_results['pixel_acc'] * 100:.2f}%"
            )
        print(status)

        current_mIoU = val_results["mIoU"] if val_results else 0.0

        save_checkpoint(
            model, optimizer, scheduler, epoch, best_mIoU,
            save_dir / "last.pt",
        )

        if current_mIoU > best_mIoU:
            best_mIoU = current_mIoU
            if cfg.checkpoint.save_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, best_mIoU,
                    save_dir / "best.pt",
                )
            print(f"  >> new best mIoU: {best_mIoU * 100:.2f}%")

    print(f"\nTraining finished. Best mIoU: {best_mIoU * 100:.2f}%")
    print(f"Checkpoints saved to: {save_dir.resolve()}")

if __name__ == "__main__":
    main()