# Reference: Benjdira, B., Ali, A. M., & Koubaa, A. (2023). Guided frequency loss for image restoration. arXiv:2309.15563
import torch
import torch.nn as nn
import torch.nn.functional as F


class GFLLoss(nn.Module):
    """
    Guided Frequency Loss for Image Restoration.

    Components:
      1) Charbonnier component ChC = charbonnier(pred, target)**2
      2) Laplacian Pyramid component PiC = MSE(Lp(pred), Lp(target))
      3) Gradual Frequency component ThC = MSE(HPF(pred), HPF(target))

    Final loss: sqrt(ChC + PiC + ThC)

    Args:
        eps (float): small constant for Charbonnier smoothing.
        omega0 (float): initial high-pass cutoff frequency (normalized [0, 0.5]).
        omegaF (float): final high-pass cutoff frequency.
        num_epochs (int): total number of training epochs.
        num_stages (int): number of frequency-band stages.
        static (bool): if True, use fixed-stage scheduling; else dynamic.
    """

    def __init__(
        self,
        eps: float = 1e-3,
        omega0: float = 0.1,
        omegaF: float = 0.5,
        num_epochs: int = 100,
        num_stages: int = 5,
        static: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.omega0 = omega0
        self.omegaF = omegaF
        self.num_epochs = num_epochs
        self.num_stages = num_stages
        self.static = static

        # Precompute stage thresholds
        self.thresholds = torch.linspace(omega0, omegaF, steps=num_stages).tolist()

    def charbonnier(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Charbonnier loss between x and y."""
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

    def laplacian1(self, x: torch.Tensor) -> torch.Tensor:
        """
        First-level Laplacian residual: downsample → upsample → subtract.
        Uses average pooling as Gaussian downsampling approximation.
        """
        down = F.avg_pool2d(x, kernel_size=2, stride=2)
        up = F.interpolate(
            down, scale_factor=2, mode="bilinear", align_corners=False
        )
        return x - up

    def high_pass(self, x: torch.Tensor, thr: float) -> torch.Tensor:
        """
        High-pass filter via FFT:
        fft → fftshift → mask → ifftshift → ifft
        """
        X = torch.fft.fftn(x, dim=(-2, -1))
        Xs = torch.fft.fftshift(X, dim=(-2, -1))
        b, c, h, w = x.shape

        # Create normalized frequency grid [-0.5, 0.5)
        u = torch.linspace(-0.5, 0.5, h, device=x.device).view(1, 1, h, 1)
        v = torch.linspace(-0.5, 0.5, w, device=x.device).view(1, 1, 1, w)
        radius = torch.sqrt(u * u + v * v)

        # Mask out frequencies below threshold
        mask = (radius >= thr).float()
        Xs_hp = Xs * mask
        Xi = torch.fft.ifftshift(Xs_hp, dim=(-2, -1))
        x_hp = torch.fft.ifftn(Xi, dim=(-2, -1)).real
        return x_hp

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, epoch: int = None
    ) -> torch.Tensor:
        """
        Compute GFL loss.

        Args:
            pred (Tensor): restored image [B, C, H, W]
            target (Tensor): ground truth image [B, C, H, W]
            epoch (int, optional): current epoch index (1-based)
        """
        # 1) Charbonnier component (squared)
        ch = self.charbonnier(pred, target) ** 2

        # 2) Laplacian Pyramid component
        lp_pred = self.laplacian1(pred)
        lp_tgt = self.laplacian1(target)
        pi = F.mse_loss(lp_pred, lp_tgt, reduction="mean")

        # 3) Gradual Frequency component
        if epoch is None:
            thr = self.omegaF
        else:
            # Static scheduling by epoch stages
            stage_len = max(1, self.num_epochs // self.num_stages)
            idx = min((epoch - 1) // stage_len, self.num_stages - 1)
            thr = self.thresholds[idx]

        th_pred = self.high_pass(pred, thr)
        th_tgt = self.high_pass(target, thr)
        th = F.mse_loss(th_pred, th_tgt, reduction="mean")

        # Combine and return
        loss = torch.sqrt(ch + pi + th + 1e-12)
        return loss
