# model_2d_resnet.py
"""
ResNet18 backbone for 2D HSI patch classification.

- 支援任意 in_channels（例如 32 個選 band）
- 可選擇是否載入 ImageNet 預訓練權重
"""

'''
(a) [sample]
一筆sample = (1, c, h, w)
           batch size, bands數, 空間尺寸
(b) [backbone]
convolution
batch normalization
activation
residual block * n
-> feature map (1, c', h', w')

(c) [global average pooling (GAP)]
-> fixed large (1, c')

(d) [Fully connected]
(1, c') -> FC -> (1, num_classes)

'''

import torch
import torch.nn as nn
import torchvision.models as models


class HSIResNet18(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 pretrained: bool = True):
        """
        Parameters
        ----------
        in_channels : int
            輸入通道數
        num_classes : int
            分類類別數
        pretrained : bool
            是否載入 ImageNet 預訓練權重
        """
        super().__init__()

        # 1)  ResNet18
        if pretrained:
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            base = models.resnet18(weights=None)

        # 2) 替換第一層 conv，支援 in_channels != 3
        old_conv = base.conv1
        if in_channels != old_conv.in_channels:
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )

            # 權重初始化
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

            if pretrained:
                # 預訓練
                with torch.no_grad():
                    c = min(3, in_channels)
                    new_conv.weight[:, :c, :, :] = old_conv.weight[:, :c, :, :]

            base.conv1 = new_conv

        # 3) 替換最後一層全連接
        in_feat = base.fc.in_features
        base.fc = nn.Linear(in_feat, num_classes)

        self.backbone = base

    def forward(self, x):
        """
        x : (B, C, H, W)
        return : (B, num_classes) 的 logit
        """
        return self.backbone(x)
