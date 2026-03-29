import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.sparse as sparse

class Block(nn.Module):
    """ConvNeXtV2 Block (NCHW, 1x1 Conv, GRN2d version)"""
    def __init__(self, dim):
        super().__init__()

        self.dwconv = sparse.SparseConv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = sparse.SparseLayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = sparse.SparseConv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = sparse.SparseGRN2d(4 * dim)
        self.pwconv2 = sparse.SparseConv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)

        mask =  sparse._get_active_ex_or_ii(H=x.shape[2], W=x.shape[3], returning_active_ex=True)    # (BCHW) *= (B1HW), mask the output of conv
        x *= mask

        x = self.grn(x)
        x = self.pwconv2(x)
        x = shortcut + x
        return x


class ConvNeXtV2_MAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768]
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims

        self.downsample_layers = nn.ModuleList()

        self.stem = nn.Sequential(
            sparse.SparseConv2d(in_channels, dims[0], kernel_size=4, stride=4),
            sparse.SparseLayerNorm2d(dims[0], eps=1e-6),
        )
        self.downsample_layers.append(self.stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                sparse.SparseLayerNorm2d(dims[i], eps=1e-6),
                sparse.SparseConv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]


    def forward(self, x):
        _, _, h, w = x.shape
        #mask = sparse._get_active_ex_or_ii(h, w, returning_active_ex=True)
        #x = x * mask

        outputs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outputs.append(x)
        return outputs


def convnextv2_mae_atto(**kwargs):
    return ConvNeXtV2_MAE(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)


def convnextv2_mae_femto(**kwargs):
    return ConvNeXtV2_MAE(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)


def convnextv2_mae_pico(**kwargs):
    return ConvNeXtV2_MAE(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)


def convnextv2_mae_nano(**kwargs):
    return ConvNeXtV2_MAE(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)


def convnextv2_mae_tiny(**kwargs):
    return ConvNeXtV2_MAE(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


def convnextv2_mae_base(**kwargs):
    return ConvNeXtV2_MAE(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


def convnextv2_mae_large(**kwargs):
    return ConvNeXtV2_MAE(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


def convnextv2_mae_huge(**kwargs):
    return ConvNeXtV2_MAE(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)