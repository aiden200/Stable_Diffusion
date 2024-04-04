import torch
from torch import nn
from torch.nn import functional as F
from utils.transformer_blocks import Self_Attention
 
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init()
        self.group_norm_1 = nn.GroupNorm(32, in_channels)
        self.conv1 - nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 = nn.GroupNorm(32, out_channels)
        self.conv2 - nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.resid_layer = nn.Identity()
        else:
            self.resid_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, I_C, H, W)

        resid = self.resid_layer(x)

        x = self.group_norm_1(x)
        x = self.conv1(x)
        F.silu(x)

        x = self.group_norm_2(x)
        x = self.conv2(x)
        F.silu(x)

        return x + resid
    

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = Self_Attention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        resid = x
        B, C, H, W = x.shape
        x = x.view(B, C, H*W)
        x = x.transpose(-1, -2)

        x = self.attention(x)
        x = x.transpose(-1, -2)

        x = x.view((B, C, H, W))

        out = x + resid

        return out
