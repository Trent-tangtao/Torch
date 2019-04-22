import torchvision as tv
import torchvision.transforms as transforms
import torch
from torchvision.transforms import ToPILImage
show = ToPILImage()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5,0.5), (0.5, 0.5, 0.5)),
                               ])

trainset = tv.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloder = torch.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)
testset = tv.datasets.CIFAR10(
    './data',
    train=False,
    download=True,
    transform=transform
)
testLoder = torch.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
(data,label) = trainset[100]
# print(classes[label])
# (data + 1) / 2是为了还原被归一化的数据
# show((data + 1) / 2).resize((100, 100)).show()

dataiter = iter(trainloder)
images, labels = dataiter.next()
# print(' '.join('11s'% classes[labels[j]] for j in range(4)))
# show(tv.utils.make_grid((images+1)/2).resize(400.100)).show()


import torch.nn as nn
import torch.nn.functional as F

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Lenet()
# print(net)


from torch import optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr =0.01, momentum=0.9)

for epoch in range(2):

    running_loss = 0.0
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，
    # 同时列出数据和数据下标，一般用在 for 循环当中。
    for i,data in enumerate(trainloder, 0):
        input, labels = data
        optimizer.zero_grad()
        outputs = net(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        if i%2000 == 1999:
            print('[%d, %5d] loss: %.3f' \
                  % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0
print(" Train Finish")




dataiter =iter(testLoder)
images, labels =dataiter.next()
print('实际的label: ', ' '.join(\
            '%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images / 2 - 0.5)).resize((400,100))

# 计算图片在每个类别上的分数
outputs = net(images)
# 得分最高的那个类
_, predicted = torch.max(outputs.data, 1)
print('预测结果: ', ' '.join('%5s'\
            % classes[predicted[j]] for j in range(4)))




correct = 0 # 预测正确的图片数
total = 0 # 总共的图片数
# 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
with torch.no_grad():
    for data in testLoder:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))