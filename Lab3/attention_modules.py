import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


def sinusoidal_2d_pos_embed(h: int, w: int, dim: int) -> torch.Tensor:
    """
    Generate fixed 2D sinusoidal positional embeddings.

    Args:
        h: Height of the feature map.
        w: Width of the feature map.
        dim: Embedding dimension (must be divisible by 4).

    Returns:
        Tensor of shape (h*w, dim) containing positional embeddings.
    """
    if dim % 4 != 0:
        raise ValueError("`dim` must be divisible by 4.")
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)
    omega = 1.0 / (10000 ** omega)
    sin_x = torch.sin(x.flatten()[:, None] * omega[None, :])
    cos_x = torch.cos(x.flatten()[:, None] * omega[None, :])
    sin_y = torch.sin(y.flatten()[:, None] * omega[None, :])
    cos_y = torch.cos(y.flatten()[:, None] * omega[None, :])
    return torch.cat([sin_x, cos_x, sin_y, cos_y], dim=1)


class GatedFeatureFusion(nn.Module):
    """
    Fuse multi-scale feature maps via learned per-scale weights.

    1) Upsample all feature maps to a common size.
    2) Global-average pool each into (N, C).
    3) Generate per-scale weights with 1×1 Conv1d + Softmax.
    4) Compute weighted sum to produce fused feature map.
    """

    def __init__(self, channels: int, num_scales: int,
                 target_size: tuple = None):
        super().__init__()
        self.num_scales = num_scales
        self.target_size = target_size
        self.gate_conv = nn.Conv1d(num_scales, num_scales, kernel_size=1,
                                   bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        n, c = feats[0].shape[:2]
        if self.target_size:
            h_t, w_t = self.target_size
        else:
            h_t, w_t = feats[0].shape[2:]

        # 1) Upsample to common resolution
        ups = [
            F.interpolate(x, size=(h_t, w_t), mode="bilinear",
                          align_corners=False)
            for x in feats
        ]
        # 2) Global-average pooling → (n, c) per scale
        pooled = torch.stack(
            [F.adaptive_avg_pool2d(x, 1).view(n, c) for x in ups],
            dim=2
        )
        # 3) Compute gates via Conv1d + Softmax over scales
        gates = self.gate_conv(pooled.transpose(1, 2))
        gates = self.softmax(gates)
        # 4) Weighted sum of upsampled features
        fused = sum(
            gates[:, i, :].view(n, c, 1, 1) * ups[i]
            for i in range(self.num_scales)
        )
        return fused


class FPNTransformer(nn.Module):
    """
    Transformer-augmented neck for FPN.

    1) Pool each FPN level to r×r tokens.
    2) Add positional and level embeddings.
    3) Encode via TransformerEncoder.
    4) Reproject tokens back to feature maps + residual.
    5) Fuse scales via GatedFeatureFusion and distribute.
    """

    def __init__(self, in_channels: int, num_heads: int = 8,
                 num_layers: int = 2, reduced_resolution: int = 14,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 **kwargs):
        super().__init__()
        d_model = in_channels
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer,
                                            num_layers=num_layers)
        self.r_res = reduced_resolution
        self.pre_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.post_proj = nn.Conv2d(d_model, in_channels, kernel_size=1)
        # Learnable level embeddings for each FPN scale
        self.level_embed = nn.Parameter(torch.randn(5, 1, d_model))
        self.fuser = GatedFeatureFusion(in_channels, num_scales=5)

    def forward(self, feats: list[torch.Tensor]):
        n = feats[0].size(0)
        seqs, shapes = [], []
        # 1) Tokenize each scale + add embeddings
        for lvl, x in enumerate(feats):
            _, c, h, w = x.shape
            shapes.append((h, w))
            p = F.adaptive_avg_pool2d(x, self.r_res)
            p = self.pre_proj(p)
            tokens = p.flatten(2).transpose(1, 2)
            pos = sinusoidal_2d_pos_embed(self.r_res, self.r_res,
                                          c).to(x.device)
            tokens = tokens + pos.unsqueeze(0) + self.level_embed[lvl]
            seqs.append(tokens)

        # 2) Transformer encoding
        seq = torch.cat(seqs, dim=1)
        seq = self.encoder(seq)

        # 3) Reconstruct per-scale features
        outs, idx = [], 0
        length = self.r_res * self.r_res
        for h, w in shapes:
            toks = seq[:, idx:idx + length, :]
            idx += length
            feat = toks.transpose(1, 2).view(n, -1, self.r_res,
                                             self.r_res)
            feat = self.post_proj(feat)
            feat = F.interpolate(feat, size=(h, w),
                                 mode="bilinear",
                                 align_corners=False)
            outs.append(feat + feats[len(outs)])

        # 4) Fuse scales globally
        fused = self.fuser(outs)

        # 5) Broadcast fused back to each scale
        final_outs = []
        for (h, w), out in zip(shapes, outs):
            fused_i = F.interpolate(fused, size=(h, w),
                                    mode="bilinear",
                                    align_corners=False)
            final_outs.append(out + fused_i)

        return final_outs


class ECAChannelAttn(nn.Module):
    """Channel attention using local 1-D convolution."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        if k % 2 == 0:
            k += 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k,
                              padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


class SpatialWindowAttn(nn.Module):
    """Spatial attention via deformable convolution."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.offset_conv = nn.Conv2d(
            channels, 2 * kernel_size * kernel_size,
            kernel_size=kernel_size, padding=pad, bias=True
        )
        self.deform_conv = DeformConv2d(
            channels, channels, kernel_size=kernel_size,
            padding=pad, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        off = self.offset_conv(x)
        y = self.deform_conv(x, off)
        y = self.bn(y)
        return x * self.sigmoid(y)


class FeedForward(nn.Module):
    """Lightweight pointwise feed-forward."""

    def __init__(self, channels: int, expansion: int = 4,
                 drop: float = 0.):
        super().__init__()
        hid = channels * expansion
        self.block = nn.Sequential(
            nn.Conv2d(channels, hid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hid, channels, kernel_size=1, bias=False),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialChannelTransformer(nn.Module):
    """
    Combined spatial and channel attention block for Mask Head.

    Steps:
    1) Channel recalibration via ECA.
    2) Spatial attention via deformable convolution.
    3) BatchNorm + residual connection.
    4) Lightweight feed-forward + residual.
    """

    def __init__(self, in_channels: int, reduction: int = 16,
                 spatial_kernel: int = 3):
        super().__init__()
        self.ca = ECAChannelAttn(in_channels)
        self.sa = SpatialWindowAttn(in_channels,
                                    kernel_size=spatial_kernel)
        self.norm = nn.BatchNorm2d(in_channels)
        self.ffn = FeedForward(in_channels, expansion=2, drop=0.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.ca(x)
        x = self.sa(x)
        x = self.norm(x)
        x = x + res
        x = x + self.ffn(x)
        return x
