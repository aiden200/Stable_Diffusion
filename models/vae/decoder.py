import torch
from torch import nn
from torch.nn import functional as F
from utils.transformer_blocks import SelfAttention
 
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm_1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm_2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

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
        self.attention = SelfAttention(1, channels)
    
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


class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (B, 512, H/8, W/8) -> (B, 512, H/4, W/4)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            # (B, 512, H/4, W/4) -> (B, 512, H/2, W/2)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            # (B, 512, H/2, W/2) -> (B, 512, H, W)
            nn.Upsample(scale_factor=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            nn.GroupNorm(32, 128), #by 32 channels each group
            nn.SELU(),

            nn.Conv2d(128, 3, kernel_size=3, padding=1)         
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, H/8, W/8)
        x /= 0.18215

        for layer in self:
            x = layer(x)
        
        return x