import torch
from torch import nn
import torch.nn.functional as F
from utils.transformer_blocks import SelfAttention


class CLIPEmbedding(nn.Modudle):
    def __init__(self, n_vocab, e, n_tokens):
        super().__init__()
        self.token_embed = nn.Embedding(n_vocab, e)
        self.positional_embedding = nn.Parameter(n_tokens, e) #Parameter just means we cache the variable


    def forward(self, x):
        embed = self.token_embed(x)
        pe = self.positional_embedding(x)
        out = embed + pe
        return out
    

class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embed):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(n_embed)
        self.attention1 = SelfAttention(n_head, n_embed)
        self.layernorm2 = nn.LayerNorm(n_embed)
        self.lin1 = nn.Linear(n_embed, 4*n_embed) # in paper
        self.lin2 = nn.Linear(4*n_embed, n_embed)
    
    def forward(self, x):
        resid = x
        x = self.layernorm1(x)
        x = self.attention1(x, mask=True) # decoder
        x += resid
        resid = x
        x = self.layernorm2(x)
        x = self.lin1(x)
        x = nn.GELU(x)
        x = self.lin2(x)
        x += resid
        return x

class CLIP(nn.Module):
    def __int__(self):
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12,768) for i in range(12)
        ])

        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens: torch.LongTensor)->torch.FloatTensor:
        tokens = tokens.type(torch.long)
        # (B, Seq_len) -> (B, S_len, E)

        s = self.embedding(tokens)
        for layer in self.layers:
            s = layer(s)
        
        #(B, Seq_len, E)
        out = self.layernorm(s)

        return out