import math

import torch
from numpy.distutils.command.config import config
from torch import nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN


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
        return x



# def load_checkpoint(model, checkpoint):

class GPT(nn.Module):
    """ GPT """
    @staticmethod
    def get_default_config():
        C = CN()
        C.model_type = 'gpt'
        # (n_layer, n_head, n_embed)
        C.n_layer = None
        C.n_head = None
        C.n_embed = None
        C.vocab_size = None
        C.block_size = None
        # dropout 超参数
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        # 布尔值，检查属性是否被定义
        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embed is not None])
        assert type_given ^ params_given
        if type_given:
            # 根据模型类型，设置参数
            config.merge_from_dict({
                # GPT-1
                'openai-gpt': dict(n_layer=12, n_head=12, n_embed=768), # 117M params
                # GPT-2 configs
                'gpt2': dict(n_layer=12, n_head=12, n_embed=768), # 124M params
                'gpt2-medium': dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
                'gpt2-large': dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
                'gpt2-xl': dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
                # Gophers
                'gopher-44m': dict(n_layer=8, n_head=16, n_embed=512),
                # I made these tiny models up
                'gpt-mini': dict(n_layer=6, n_head=6, n_embed=192),
                'gpt-micro': dict(n_layer=4, n_head=4, n_embed=128),
                'gpt-nano': dict(n_layer=3, n_head=3, n_embed=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))





















