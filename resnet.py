# coding:utf8
from torch import nn
from torch.nn import functional as F
import torch
import functools
from torch.nn import Parameter


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride = 1, shortcut = None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias = False),
            nn.InstanceNorm2d(outchannel), # InstanceNorm2d, BatchNorm2d
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias = False),
            nn.InstanceNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(ResNet34, self).__init__()
        self.model_name = 'resnet34'

        # 前几层: 图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(inchannel, 32, 7, 1, 3, bias = False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True))

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(32, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride = 1)
        self.layer3 = self._make_layer(128, 64, 6, stride = 1)
        self.layer4 = self._make_layer(64, 32, 3, stride = 1)

        self.toImagesize = nn.Conv2d(32, 1, 3, 1, 1, bias = False)

        self.linear = nn.Linear(4096, outchannel) # 16384 4096
        #self.dropout = nn.Dropout(p = 0.2)
        self.final = nn.Sigmoid()

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.InstanceNorm2d(outchannel))

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):  # 
        x = self.pre(x)   # 

        x = self.layer1(x)  # 
        x = self.layer2(x)  # 
        x = self.layer3(x)  # 
        x = self.layer4(x)  # 

        x = self.toImagesize(x) # 
        x = x.view(x.size(0), -1) # 
        x = self.linear(x)
        #x = self.dropout(x)
        x = self.final(x) # sigmoid转化到[0, 1]中
        x = x * 2 - 1
        return x



