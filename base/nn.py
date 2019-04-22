import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(in_feature,out_feature))
        self.b = nn.Parameter(torch.randn(out_feature))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b

layer = Linear(4,3)
input = torch.randn(2,4)
output = layer(input)
print(output)
for name, params in layer.named_parameters():
    print(name,": ",params)



class Perception(nn.Module):
    def __init__(self, in_features, hidden_featues, out_features):
        super(Perception,self).__init__()
        self.layer1 = Linear(in_features, hidden_featues)
        self.layer2 = Linear(hidden_featues, out_features)
    def forward(self, x):
        x = self.layer1(x)
        x = torch.sigmoid(x)
        return self.layer2(x)

percetion = Perception(3,4,1)
for name, params in percetion.named_parameters():
    print(name,":", params.size())


from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
to_tensor = ToTensor()
to_pil = ToPILImage()
test = Image.open('test.jpg')
test.show()



# pytorch 提供了预训练好的写好的模型
import torchvision.models as models

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)

