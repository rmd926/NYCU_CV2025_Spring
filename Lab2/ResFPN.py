import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    """Residual Convolution Block."""
    def __init__(self, channels):
        super(ResidualConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = F.relu(out)
        return out


class ResFPN(nn.Module):
    """
    Enhance FPN by processing each scale with multiple ResidualConvBlocks.
    """
    def __init__(self, fpn, num_res_blocks=1):
        super(ResFPN, self).__init__()
        self.fpn = fpn
        try:
            in_channels = fpn.out_channels
        except AttributeError:
            in_channels = fpn.inner_blocks[0].out_channels
        self.num_levels = len(fpn.inner_blocks)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(*[ResidualConvBlock(in_channels)
                            for _ in range(num_res_blocks)])
            for _ in range(self.num_levels)
        ])

    def forward(self, x):
        features = self.fpn(x) 
        keys = list(features.keys())
        enhanced_features = {}
        for idx, key in enumerate(keys):
            feat = features[key]
            if idx < len(self.res_blocks):
                feat = self.res_blocks[idx](feat)
            enhanced_features[key] = feat
        return enhanced_features
