import math

import torch
from numpy.distutils.command.config import config
from torch import nn
from torch.nn import functional as F


class NewGELU(nn.Module):
    """
    GELU 激活函数
    """
    def fowrard(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # q, k, v 的 projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # outer projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # dropout
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch_size, sequence_length, embedding_size

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, hs)
        v = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, hs)

        # self-attention: (B, n_head, T, C) x (B, n_head, C, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, n_head, T, T) x (B, n_head, T, hs) -> (B, n_head, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output 映射
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """
    Transformer 块
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc = nn.Linear(config.n_embd, 4 * config.n_embd),
                c_proj = nn.Linear(4 * config.n_embd, config.n_embd),
                act = NewGELU(),
                dropout = nn.Dropout(config.resid_pdrop)
            )
        )
        self.mlpf = nn.Sequential(
            *self.mlp.values()
        )

    def forward(self, x):
        # gpt2是先layernorm，原生Transformer是后layernorm
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))

# def load_checkpoint(model, checkpoint):