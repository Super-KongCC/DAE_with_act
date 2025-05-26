import torch
from torch import nn
import math
import numpy as np
from collections import OrderedDict



class sin(nn.Module):
    def __init__(self):
        super(sin,self).__init__()
    def forward(self,x):
        return torch.sin(x)

# 定义加权激活函数类
class WA(nn.Module):
    def __init__(self, activation1=sin(), activation2=nn.LogSigmoid()):
        super(WA, self).__init__()
        assert callable(activation1) and callable(activation2), "激活函数必须是可调用的"
        self.activation1 = activation1
        self.activation2 = activation2
        self.weight = nn.Parameter(torch.tensor([0.5]))  # 初始权重为0.5

    def forward(self, x):
        out1 = self.activation1(x)
        out2 = self.activation2(x)
        weight = self.weight
        return weight * out1 + (1 - weight) * out2
