import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from model import PromptIR
from dataloader import get_dataloader
from tqdm import tqdm
from utils import plot_training_curves
from GFL_Loss import GFLLoss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def compute_psnr(mse):
    return 10 * torch.log10(1.0 / mse)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Charbonnier(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, out, tgt):
        diff = out - tgt
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


def train_one_epoch(model, loader, criterion, optimizer, device,
                    scaler=None, accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    loop = tqdm(loader, desc='Train', leave=False)
    for i, (deg, clean, _) in enumerate(loop):
        # simple flips/rotations
        if random.random() < 0.5:
            deg, clean = torch.flip(deg, dims=[3]), torch.flip(clean, dims=[3])
        if random.random() < 0.5:
            deg, clean = torch.flip(deg, dims=[2]), torch.flip(clean, dims=[2])
        k = random.randint(0, 3)
        if k:
            deg = torch.rot90(deg, k, dims=[2, 3])
            clean = torch.rot90(clean, k, dims=[2, 3])

        deg, clean = deg.to(device), clean.to(device)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            preds = model(deg)
            loss = criterion(preds, clean) / accum_steps

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
        loop.set_postfix(loss=total_loss / ((i + 1) * deg.size(0)))

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    loop = tqdm(loader, desc='Val', leave=False)
    with torch.no_grad():
        for deg, clean, _ in loop:
            deg, clean = deg.to(device), clean.to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(deg)
                loss = criterion(preds, clean)
            total_loss += loss.item() * deg.size(0)

            mse = nn.functional.mse_loss(preds, clean,
                                         reduction='none')
            mse = mse.view(deg.size(0), -1).mean(dim=1)
            total_psnr += compute_psnr(mse).sum().item()
            loop.set_postfix(loss=loss.item(),
                             psnr=total_psnr / len(loader.dataset))

    return total_loss / len(loader.dataset), total_psnr / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='path to training/validation data')
    parser.add_argument('--checkpoint', required=True,
                        help='pretrained checkpoint (.pth)')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--accum_steps', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--save_dir', default='finetune_ckpt',
                        help='directory to save checkpoints')
    parser.add_argument('--fp16', action='store_true',
                        help='enable mixed precision training')
    parser.add_argument('--loss', choices=['charbonnier', 'gfl'],
                        default='gfl',
                        help='select loss function')
    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = args.fp16 and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    train_loader = get_dataloader(args.data_dir, 'train',
                                  args.batch_size, args.num_workers)
    val_loader = get_dataloader(args.data_dir, 'val',
                                args.batch_size, args.num_workers)

    model = PromptIR(decoder=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint), strict=False)

    # choose loss
    if args.loss == 'charbonnier':
        criterion = Charbonnier(eps=1e-3)
    else:
        criterion = GFLLoss(
            eps=1e-3,
            omega0=0.1,
            omegaF=0.5,
            num_epochs=args.epochs,
            num_stages=5,
            static=True
        )

    # optimizer with weight decay split
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('.bias') or 'norm' in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = AdamW([
        {'params': decay, 'weight_decay': 1e-4},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=args.lr)

    # schedulers
    warmup_sched = LinearLR(
        optimizer,
        start_factor=5e-6,
        end_factor=1.0,
        total_iters=args.warmup
    )
    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup,
        eta_min=5e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup]
    )

    history = {'train_loss': [], 'val_loss': [], 'val_psnr': []}

    try:
        for ep in range(1, args.epochs + 1):
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {ep}/{args.epochs}  LR={lr:.2e}  Loss={args.loss}")

            tl = train_one_epoch(model, train_loader, criterion,
                                 optimizer, device, scaler,
                                 accum_steps=args.accum_steps)
            vl, vp = validate(model, val_loader, criterion,
                              device, use_amp)

            history['train_loss'].append(tl)
            history['val_loss'].append(vl)
            history['val_psnr'].append(vp)

            fname = (f"ep{ep:03d}_{args.loss}_"
                     f"vl{vl:.4f}_psnr{vp:.4f}.pth")
            torch.save(model.state_dict(),
                       os.path.join(args.save_dir, fname))

            print(f"Train L={tl:.4f} | Val L={vl:.4f} "
                  f"| PSNR={vp:.4f} dB")

    except KeyboardInterrupt:
        print("\nInterrupted.")

    plot_training_curves(history)


if __name__ == '__main__':
    main()
