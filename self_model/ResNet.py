import torch
import torchvision
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):

    def __init__(self, inchannel, outchanel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchanel, 3,stride, 1, bias=False),
            nn.BatchNorm2d(outchanel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchanel, outchanel, 3, 1, 1,bias=False),
            nn.BatchNorm2d(outchanel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet,self).__init__()
        self.pre=nn.Sequential(
            nn.Conv2d(3, 64,7,2,3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,2,1)
        )
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(63, 128, 4,stride=2)
        self.layer3 = self._make_layer(128,256,6,stride=2)
        self.layer4 = self._make_layer(256,512 ,3 ,stride=2)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut=nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1,stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = Resnet()
input = torch.randn(1,3,224,224)
out= model(input)

