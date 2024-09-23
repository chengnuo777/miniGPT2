import math

import torch
from torch import nn


class NewGELU(nn.Module):
    """
    GELU 激活函数
    """
    def fowrard(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


