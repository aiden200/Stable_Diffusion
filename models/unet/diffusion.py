import torch
from torch import nn
from torch.nn import functional as F
from utils.transformer_blocks import SelfAttention, CrossAttention


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_positional_encoding = TimeEmbedding(320)
        self.unet = Unet()
        self.final_layer = Unet_Out(320, 4)

    def forward(self, z: torch.Tensor, prompt_embed: torch.Tensor, time: torch.Tensor):
        # z: (B, 4, H/8, W/8)
        # prompt_embed: (B, Seq_len, E)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        # just tells what point of time the tensor is in
        time = self.time_positional_encoding(time)

        # converts back to original shape
        z_prime = self.unet(z, prompt_embed, time)

        out = self.final_layer(z_prime)

        # (B, 4, H/8, W/8)
        return out
    
class Unet_Attn_block(nn.Module):
    def __init__(self, n_head, embed, context=768):
        super().__init__()
        channels = n_head*embed
        self.groupNorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attn1 = SelfAttention(n_head, channels, in_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attn2 = CrossAttention(n_head, channels, context, in_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.lin1 = nn.Linear(channels, 8*channels)
        self.lin2 = nn.Linear(4*channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, prompt_embed):
        # x (B, C, H, W), prompt_embed: (B, Seq_len, E)
        resid = x
        x = self.groupNorm(x)
        x = self.conv(x)

        B, C, H, W = x.shape
        x = x.view((B, C, H*W))
        x = x.transpose(-1,-2)

        resid_short = x
        x = self.layernorm_1(x)
        x = self.attn1(x)
        x += resid_short

        resid_short = x
        x = self.layernorm_2(x)
        x = self.attn2(x, prompt_embed) # cross attention with prompt embeddings
        x, g = self.lin1(x).chunk(2,dim=-1)
        x = x * F.gelu(g)
        x = self.lin2(x)
        x += resid_short

        x = x.transpose(-1, -2)
        x = x.view((B, C, H, W))
        out = self.conv2(x) + resid

        return out

    
class Unet_resid_block(nn.Module):
    def __init__(self, in_c, out_c, time_e=1280):
        super().__init__()
        self.groupNorm = nn.GroupNorm(32, in_c)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.lin_time = nn.Linear(time_e, out_c)
        self.groupNorm2 = nn.GroupNorm(32, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)

        if in_c == out_c:
            self.resid_layer = nn.Identity()
        else:
            self.resid_layer = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
    
    def forward(self, x, time):
        # x: (B, in_c, H, W), time: (1, 1280)
        resid = x
        x = self.groupNorm(x)
        x = F.silu(x)
        x = self.conv(x)
        time = F.silu(time)
        time = self.lin_time(time)
        x = x + time.unsqueeze(-1).unsqueeze(-1) # (1, 1280, 1, 1)
        x = self.groupNorm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        out = x + self.resid_layer(resid)
        return out


class SwitchSequential(nn.Sequential):

    def forward(self, x, prompt_embed, time):
        for layer in self:
            if isinstance(layer, Unet_Attn_block):
                x = layer(x, prompt_embed)
            elif isinstance(layer, Unet_resid_block):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        out = self.conv(x)
        return out


class Unet_Out(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.groupNorm = nn.GroupNorm(32, in_c)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

    def forward(self, x):
        # (B, 320, H/8, W/8)
        x = self.groupNorm(x)
        x = F.silu(x)
        out = self.conv(x)
        # (B, 4, H/8, W/8)
        return out


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (B, 4, H/8, W/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(Unet_resid_block(320, 320), Unet_Attn_block(8, 40)),
            SwitchSequential(Unet_resid_block(320, 320), Unet_Attn_block(8, 40)),
            # (B, 320, H/8, W/8)->(B, 640, H/16, W/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(Unet_resid_block(320, 640), Unet_Attn_block(8, 80)),
            SwitchSequential(Unet_resid_block(640, 640), Unet_Attn_block(8, 80)),
            # (B, 640, H/16, W/16)-> (B, 1280, H/32, W/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(Unet_resid_block(640, 1280), Unet_Attn_block(8, 160)),
            SwitchSequential(Unet_resid_block(1280, 1280), Unet_Attn_block(8, 160)),
            # (B, 1280, H/32, W/32)-> (B, 1280, H/64, W/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(Unet_resid_block(1280, 1280)),
            SwitchSequential(Unet_resid_block(1280, 1280))
        ])

        self.unet_tail = SwitchSequential(
            Unet_resid_block(1280, 1280),
            Unet_Attn_block(8, 160),
            Unet_resid_block(1280, 1280)
        )

        # Since we are adding the skip connection (the encoder outputs), so we double what we need
        self.decoders = nn.ModuleList([
            SwitchSequential(Unet_resid_block(2560, 1280)),
            SwitchSequential(Unet_resid_block(2560, 1280)),
            SwitchSequential(Unet_resid_block(2560, 1280), Upsample(1280)),
            SwitchSequential(Unet_resid_block(2560, 1280),
                             Unet_Attn_block(8, 160)),
            SwitchSequential(Unet_resid_block(2560, 1280),
                             Unet_Attn_block(8, 160)),
            SwitchSequential(Unet_resid_block(1920, 1280),
                             Unet_Attn_block(8, 160), Upsample(1280)),
            SwitchSequential(Unet_resid_block(1920, 640),
                             Unet_Attn_block(8, 80)),
            SwitchSequential(Unet_resid_block(1280, 640),
                             Unet_Attn_block(8, 80)),
            SwitchSequential(Unet_resid_block(960, 640),
                             Unet_Attn_block(8, 80), Upsample(640)),
            SwitchSequential(Unet_resid_block(960, 320),
                             Unet_Attn_block(8, 40)),
            SwitchSequential(Unet_resid_block(640, 320),
                             Unet_Attn_block(8, 40)),
            SwitchSequential(Unet_resid_block(640, 320),
                             Unet_Attn_block(8, 40)),
        ])
    def forward(self, x, context, time):
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.unet_tail(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class TimeEmbedding(nn.Module):
    def __init__(self, e):
        super().__init__()
        self.lin1 = nn.Linear(e, 4*e)
        self.lin2 = nn.Linear(4*e, 4*e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1, 320)
        x = self.lin1(x)
        x = F.silu(x)
        out = self.lin2(x)
        # (1, 1280)
        return out
