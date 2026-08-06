# model_3d_resnet18.py
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class HSIResNet3D18(nn.Module):
    """
    3D ResNet-18 for HSI patch classification (Route A: Conv3d)
    Input : (B, 1, D, H, W)  e.g. (B,1,32,128,128)
    Output: (B, num_classes)
    """

    def __init__(self, num_classes=2, in_channels=1, pretrained=False):
        super().__init__()
        if in_channels != 1:
            raise ValueError("This implementation assumes dataset outputs (B,1,D,H,W). Keep in_channels=1.")

        if pretrained:
            weights = R3D_18_Weights.DEFAULT
            self.backbone = r3d_18(weights=weights)
        else:
            self.backbone = r3d_18(weights=None)

        old_conv = self.backbone.stem[0]
        new_conv = nn.Conv3d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                w = old_conv.weight  # tensor
                new_conv.weight.copy_(w.mean(dim=1, keepdim=True))

        self.backbone.stem[0] = new_conv

        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        return self.backbone(x)
