import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.datasets import CIFAR10


def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, 3, stride=stride, padding=1, bias=False)


class residual_block(nn.Module):
    def __init__(self, in_channel, out_channel, same_shape=True):
        super(residual_block, self).__init__()
        self.same_shape=same_shape

        strides = 1 if self.same_shape else 2

        self.conv1 = conv3x3(in_channel, out_channel, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)

        if not self.same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel,1, stride=strides)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out), True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), True)

        if not self.same_shape:
            x = self.conv3(x)

        return F.relu(x+out, True)



class resnet(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(resnet, self).__init__()
        self.verbose = verbose

        self.block1 = nn.Conv2d(in_channel, 64, 7,2)

        self.block2 = nn.Sequential(
            nn.MaxPool2d(3,2),
            residual_block(64,64),
            residual_block(64,64)
        )

        self.block3 = nn.Sequential(
            residual_block(64,128, False),
            residual_block(128,128)
        )
        self.block4 = nn.Sequential(
            residual_block(128,256, False),
            residual_block(256,256)
        )
        self.block5 = nn.Sequential(
            residual_block(256, 512,False),
            residual_block(512,512),
            nn.AvgPool2d(3)
        )

        self.classifier = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
