import  torch
import torch.nn as nn
from torch.nn import functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, **kwargs)
        self.bn = nn.BatchNorm2d(out_channel, eps=0.001)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channel, 64, kernel_size=1)
        self.branch5x5_1 = BasicConv2d(in_channel, 48, Kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3_1 = BasicConv2d(in_channel, 64, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(96, 64, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channel, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=2, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, 1)
