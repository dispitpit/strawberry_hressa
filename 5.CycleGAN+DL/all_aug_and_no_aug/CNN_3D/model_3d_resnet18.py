import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class HSIResNet3D18(nn.Module):
    """
    3D ResNet18 for HSI classification

    Input shape:
        (B, 1, D, H, W)
    where:
        D = selected spectral bands (e.g., 32)

    與 2D ResNet18 的差別：
    - 2D: (B, C, H, W), C=32，Conv2d 只在空間平面滑動
    - 3D: (B, 1, D, H, W)，Conv3d 會在光譜與空間一起滑動
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 2, pretrained: bool = False):
        super().__init__()

        if pretrained:
            try:
                base = r3d_18(weights=R3D_18_Weights.DEFAULT)
            except Exception:
                base = r3d_18(weights=None)
        else:
            base = r3d_18(weights=None)

        old_conv = base.stem[0]
        if in_channels != old_conv.in_channels:
            new_conv = nn.Conv3d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            if pretrained and old_conv.weight.shape[1] >= 1:
                with torch.no_grad():
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                    new_conv.weight.copy_(mean_w)

            base.stem[0] = new_conv

        in_feat = base.fc.in_features
        base.fc = nn.Linear(in_feat, num_classes)
        self.backbone = base

    def forward(self, x):
        return self.backbone(x)
