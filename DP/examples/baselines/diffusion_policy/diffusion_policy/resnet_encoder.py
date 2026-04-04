"""Dual-stream shared-weight ResNet-18 visual encoder."""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    """Encode 6-channel (base + hand camera) images via a shared ResNet-18.

    Input:  (B, 6, H, W) -- two 3-channel images concatenated along channel dim
    Output: (B, out_dim)
    """

    def __init__(self, out_dim: int = 512, pretrained: bool = True):
        super().__init__()
        self.out_dim = out_dim

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Remove final FC and avgpool -- we apply our own global avg pool
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ResNet-18 layer4 outputs 512 channels; two streams -> 1024
        self.fc = nn.Linear(512 * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 6, H, W) -> (B, out_dim)"""
        x1 = x[:, :3]  # base camera
        x2 = x[:, 3:]  # hand camera

        # Batch both streams through the shared backbone
        both = torch.cat([x1, x2], dim=0)          # (2B, 3, H, W)
        feats = self.backbone(both)                  # (2B, 512, h, w)
        feats = self.pool(feats).flatten(1)          # (2B, 512)

        f1, f2 = feats.chunk(2, dim=0)              # each (B, 512)
        combined = torch.cat([f1, f2], dim=-1)       # (B, 1024)
        return self.fc(combined)                     # (B, out_dim)
