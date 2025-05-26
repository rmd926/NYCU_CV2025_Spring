import os
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
from tqdm import tqdm

from model import PromptIR
from dataloader import get_dataloader
from utils import plot_training_curves
from augment import apply_augment
from GFL_Loss import GFLLoss


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def compute_psnr(mse):
    return 10 * torch.log10(1.0 / mse)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Charbonnier(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, out, tgt):
        diff = out - tgt
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    augs,
    probs,
    alphas,
    aux_prob,
    aux_alpha,
    mix_p,
    scaler=None,
    accum_steps: int = 1,
):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    loop = tqdm(loader, desc="Training", leave=False)

    for i, (deg, clean, _) in enumerate(loop):
        # spatial TTA
        if random.random() < 0.5:
            deg = torch.flip(deg, dims=[3])
            clean = torch.flip(clean, dims=[3])
        if random.random() < 0.5:
            deg = torch.flip(deg, dims=[2])
            clean = torch.flip(clean, dims=[2])
        k = random.randint(0, 3)
        if k:
            deg = torch.rot90(deg, k, dims=[2, 3])
            clean = torch.rot90(clean, k, dims=[2, 3])

        # additional augmentations
        deg, clean, _, _ = apply_augment(
            deg,
            clean,
            augs,
            probs,
            alphas,
            aux_prob,
            aux_alpha,
            mix_p,
        )

        deg = deg.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            output = model(deg)
            loss = criterion(output, clean) / accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            if scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps * deg.size(0)
        loop.set_postfix(loss=(total_loss / ((i + 1) * deg.size(0))))

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device, use_amp: bool = False):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    loop = tqdm(loader, desc="Validation", leave=False)

    with torch.no_grad():
        for deg, clean, _ in loop:
            deg = deg.to(device)
            clean = clean.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(deg)
                loss = criterion(output, clean)

            total_loss += loss.item() * deg.size(0)
            mse = nn.functional.mse_loss(output, clean, reduction="none")
            mse = mse.view(deg.size(0), -1).mean(dim=1)
            total_psnr += compute_psnr(mse).sum().item()
            loop.set_postfix(
                loss=loss.item(), psnr=compute_psnr(mse).mean().item()
            )

    return total_loss / len(loader.dataset), total_psnr / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="path to dataset root"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="per-GPU batch size"
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=4,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of data loader workers"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="initial learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=250, help="total number of training epochs"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        help="number of warmup epochs",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="gfl_checkpoints",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="enable mixed precision training",
    )
    parser.add_argument(
        "--loss",
        choices=["charbonnier", "gfl"],
        default="charbonnier",
        help="select loss function",
    )
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    train_loader = get_dataloader(
        args.data_dir, "train", args.batch_size, args.num_workers
    )
    val_loader = get_dataloader(
        args.data_dir, "val", args.batch_size, args.num_workers
    )

    model = PromptIR(decoder=True).to(device)

    # choose loss
    if args.loss == "charbonnier":
        criterion = Charbonnier(eps=1e-3)
    else:
        criterion = GFLLoss(
            eps=1e-3,
            omega0=0.1,
            omegaF=0.5,
            num_epochs=args.epochs,
            num_stages=5,
            static=True,
        )

    # optimizer w/ decay split
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = AdamW(
        [
            {"params": decay, "weight_decay": 1e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    # scheduler
    warmup = LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs]
    )

    # augment settings
    augs = ["cutblur", "mixup", "blend", "none"]
    probs = [0.5, 0.5, 0.5, 1.0]
    alphas = [0.3, 0.8, 0.6, 0.0]
    aux_prob = 0.5
    aux_alpha = 0.8
    mix_p = [0.25, 0.25, 0.25, 0.25]

    history = {"train_loss": [], "val_loss": [], "val_psnr": []}

    try:
        for epoch in range(1, args.epochs + 1):
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"\nEpoch {epoch}/{args.epochs}, "
                f"LR={lr:.2e}, Loss={args.loss}"
            )

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                augs,
                probs,
                alphas,
                aux_prob,
                aux_alpha,
                mix_p,
                scaler,
                accum_steps=args.accum_steps,
            )
            val_loss, val_psnr = validate(
                model, val_loader, criterion, device, use_amp
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_psnr"].append(val_psnr)

            ckpt_name = (
                f"ep{epoch:03d}_{args.loss}"
                f"_vl{val_loss:.4f}_psnr{val_psnr:.4f}.pth"
            )
            torch.save(
                model.state_dict(), os.path.join(args.save_dir, ckpt_name)
            )

            print(
                f"Train L={train_loss:.4f} | "
                f"Val L={val_loss:.4f} | PSNR={val_psnr:.4f} dB"
            )

    except KeyboardInterrupt:
        print("\nInterrupted. Displaying curves…")

    plot_training_curves(history)


if __name__ == "__main__":
    main()
