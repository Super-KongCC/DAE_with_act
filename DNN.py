import torch
from torch import nn
import math
import numpy as np
from collections import OrderedDict
from act import sin,WA


class DNN(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            depth,
            act = WA
    ):
        super(DNN,self).__init__()
        #输入层
        layers = [('input',torch.nn.Linear(input_size,hidden_size))]
        layers.append(('input_activation',act()))
        #隐藏层
        for i in range(depth):
            layers.append(('hidden_%d' % i,torch.nn.Linear(hidden_size,hidden_size)))
            layers.append(('hidden_activation_%d' % i,act()))
        #输出层
        layers.append(('output',torch.nn.Linear(hidden_size,output_size)))

        #封装为连续的神经网络
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self,x):
        return self.layers(x)