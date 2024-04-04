import torch
from torch import nn
from torch.nn import functional as F
from models.vae.decoder import VAE_Attention_Block, VAE_Residual_Block

class VAE_encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            #(B, C, H, W) -> (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            #(B, 128, H, W) -> (B, 128, H, W)
            VAE_Residual_Block(128, 128),
            #(B, 128, H, W) -> (B, 128, H, W)
            VAE_Residual_Block(128, 128),

            #(B, 128, H, W) -> (B, 256, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            #(B, 128, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_Residual_Block(128, 256),
            #(B, 256, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_Residual_Block(256, 256),

            #(B, 256, H/2, W/2) -> (B, 512, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            #(B, 128, H/4, W/4) -> (B, 256, H/4, W/4)
            VAE_Residual_Block(256, 512),
            #(B, 512, H/4, W/4) -> (B, 512, H/4, W/4)
            VAE_Residual_Block(512, 512),

            #(B, 512, H/4, W/4) -> (B, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_Residual_Block(512, 512),
            VAE_Residual_Block(512, 512),     
            #(B, 128, H/8, W/8) -> (B, 128, H/8, W/8)

            VAE_Residual_Block(512, 512),       
            VAE_Attention_Block(512),
            VAE_Residual_Block(512,512),
            nn.GroupNorm(32, 512),
            nn.SiLU()

            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        #x: (B, C, H, W), noise: (B, O_C, H/8, W/8)
        for layer in self:
            if getattr(layer, 'stride', None) == (2,2):
                # (P_L, P_R, P_B, P_T) so we are doing asymetrical padding 
                x = F.pad(x, (0, 1, 0, 1))
            x = layer(x)