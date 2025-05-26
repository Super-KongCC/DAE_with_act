import torch
from torch import nn
import math
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import torchdiffeq
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, hidden_size, act=nn.Sigmoid):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act(),
            nn.Linear(hidden_size, hidden_size),
            act()
        )

    def forward(self, t, h):
        return self.net(h)
    
class ODEBlock(nn.Module):
    def __init__(self, hidden_size, act=nn.Tanh):  # 让 ODEBlock 直接接收 hidden_size 和 act
        super(ODEBlock, self).__init__()
        self.ode_func = ODEFunc(hidden_size, act)  # 创建 ODE 计算单元

    def forward(self, h0):
        t = torch.linspace(0, 1, 2)  # 时间范围
        h1 = odeint(self.ode_func, h0, t, method='rk4')  # 求解 ODE
        return h1[-1]  # 取最终状态

# 定义神经ODE DNN
class ODE_DNN(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            depth,
            num_ode_blocks=2,  # 控制 ODEBlock 数量
            act=nn.ReLU
    ):
        super(ODE_DNN, self).__init__()

        layers = [('input', nn.Linear(input_size, hidden_size))]
        layers.append(('input_activation', act()))

        # 记录 ODEBlock 索引
        self.ode_blocks = nn.ModuleList([ODEBlock(hidden_size, act) for _ in range(num_ode_blocks)])
        self.ode_positions = []  # 存储 ODEBlock 在 layers 里的位置

        # 交替使用普通 Linear 层和 ODEBlock
        ode_count = 0
        for i in range(depth):
            if i % 2 == 0 and ode_count < num_ode_blocks:  # 控制 ODEBlock 的数量
                layers.append((f'ode_placeholder_{i}', nn.Identity()))
                self.ode_positions.append(len(layers) - 1)  # 记录 ODEBlock 的索引
                ode_count += 1
            else:
                layers.append((f'hidden_{i}', nn.Linear(hidden_size, hidden_size)))
                layers.append((f'hidden_activation_{i}', act()))

        layers.append(('output', nn.Linear(hidden_size, output_size)))
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        ode_index = 0  # 追踪 ODEBlock 的索引
        for i, layer in enumerate(self.layers):
            if i in self.ode_positions:  # 确保 ODEBlock 对应 Identity 层
                x = self.ode_blocks[ode_index](x)
                ode_index += 1  # 更新索引
            else:
                x = layer(x)
        return x