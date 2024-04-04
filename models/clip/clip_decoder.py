import torch
from torch import nn
import torch.nn.functional as F
from utils.transformer_blocks import SelfAttentinon

class CLIP(nn.Module):
    def __int__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)