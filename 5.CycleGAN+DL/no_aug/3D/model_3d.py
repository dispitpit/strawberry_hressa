# model_3d.py
# 最基礎的模型 無用
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleHSI3DNet(nn.Module):
    """
    Input:  (B, 1, D, H, W)  where D=32
    Output: (B, num_classes)
    """
    def __init__(self, num_classes=2, base=16):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, base, kernel_size=3, padding=1),
            nn.BatchNorm3d(base),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(base, base*2, kernel_size=3, padding=1),
            nn.BatchNorm3d(base*2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(base*2, base*4, kernel_size=3, padding=1),
            nn.BatchNorm3d(base*4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d(output_size=1),   # -> (B, C,1,1,1)
        )
        self.classifier = nn.Linear(base*4, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)
