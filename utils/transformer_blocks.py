import torch
from torch import nn
from  torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, embed: int, in_bias=True, out_bias = True):
        super().__init__()
        self.qkv_w = nn.Linear(embed, 3*embed, bias=in_bias)
        self.lm_head = nn.Linear(embed, embed, bias=out_bias)
        self.n_heads = n_heads
        self.d_head = embed // n_heads
    
    def forward(self, x: torch.Tensor, mask=False):
        # x(B, len, E)
        B, L, E = x.shape

        q, k, v = self.qkv_w(x).chunk(3, dim=-1)

        q = q.view((B, L, self.n_heads, self.d_head)).transpose(1, 2)
        k = k.view((B, L, self.n_heads, self.d_head)).transpose(1, 2)
        v = v.view((B, L, self.n_heads, self.d_head)).transpose(1, 2)

        qk = q @ v.transpose(-1,-2)
        qk = qk * 1/math.sqrt(self.d_head)

        if mask:
            mask = torch.ones_like(qk, dtype=torch.bool).triu(1)
            qk.masked_fill_(mask, -torch.inf)
        
        qk = F.softmax(qk, dim=-1)

        qkv = qk @ v

        qkv.transpose(1,2) #(B, L, H, embed/H))

        qkv = qkv.reshape((B, L, E))

        output = self.lm_head(qkv)

        #(B, L, embed)
        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, embed_q, embed_cross, in_bias=True, out_bias = True):
        super().__init__()
        self.q_w = nn.Linear(embed_q, embed_q, bias=in_bias)
        self.k_w = nn.Linear(embed_cross, embed_q, bias=in_bias)
        self.v_w = nn.Linear(embed_cross, embed_q, bias=in_bias)

        self.n_heads = n_heads
        self.d_head = embed_q // n_heads

        self.lm_head = nn.Linear(embed_q, embed_q, bias=out_bias)

    def forward(self, x: torch.Tensor, cross: torch.Tensor):
        # x(B, len, E) cross (B, len_c, E_c)
        B, L, E = x.shape
        B, L_c, E_c = cross.shape

        q = self.q_w(x)
        k = self.k_w(cross)
        v = self.v_w(cross)

        q = q.view((B, -1, self.n_heads, self.d_head)).transpose(1, 2)
        k = k.view((B, -1, self.n_heads, self.d_head)).transpose(1, 2)
        v = v.view((B, -1, self.n_heads, self.d_head)).transpose(1, 2)

        qk = q @ v.transpose(-1,-2)
        qk = qk * 1/math.sqrt(self.d_head)

        qk = F.softmax(qk, dim=-1)

        qkv = qk @ v

        qkv.transpose(1,2).contiguous()

        qkv = qkv.reshape((B, L, E))

        output = self.lm_head(qkv)

        #(B, L, embed)
        return output