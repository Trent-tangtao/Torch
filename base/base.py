import torch
import numpy as np

# 1. 对比numpy 和 torch

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensortoarray = torch_data.numpy()
# print(np_data,"\n", torch_data, "\n",tensortoarray)


# abs  sin mean
data = [-1, -2, 1]
tensor = torch.FloatTensor(data)
# print(torch.abs(tensor))

# 矩阵
data = [[1,2], [3,4]]
tensor = torch.FloatTensor(data)
# print('\nnumpy',np.matmul(data,data),'\ntorch',torch.mm(tensor,tensor))
# 接受的必须是tensor的形式，注意dot的用法不同


# 2. variable 变量的形式，将tensor放到variable 里面来
#  variable 会搭建计算图，进行反向传
from torch.autograd import Variable
tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

v_out.backward()
#print(variable.grad)  # 打印梯度      反向传播的过程
#print(variable)
#print(variable.data)   # variable.data  是tensor
#print(variable.data.numpy())



# 3.激励函数必须可微，这样才能反向传播
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.linspace(-5,5,200)
x = Variable(x)
x_np = x.data.numpy()


y_relu = F.relu(x).data.numpy()
# relu , tanh ....


# 4.网络

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())
x, y = Variable(x), Variable(y)

class Net(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()   # 必要步骤,继承
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
# print(net)  直接看网络的结构!!!

plt.ion()   # 画图
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_fun = torch.nn.MSELoss()

for t in range(100):
    prediction = net(x)

    loss = loss_fun(prediction, y)

    optimizer.zero_grad()  # 先降所有参数的梯度为0
    loss.backward()   # 计算每次的梯度
    optimizer.step()  # 优化梯度

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)


plt.ioff()
plt.show()




###########################

import torch

# x = torch.Tensor([[1, 2], [3, 4]])
x = torch.rand(5, 3)
y = torch.rand(5, 3)
res = torch.Tensor(5, 3)
torch.add(x, y, out=res)
# print(res)

# 注意，函数名后面带下划线_ 的函数会修改Tensor本身。
# 例如，x.add_(y)和x.t_()会改变 x，但x.add(y)和x.t()返回一个新的Tensor， 而x不变。

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        # 卷积层 '1'表示输入图片为单通道, '6'表示输出通道数，'5'表示卷积核为5*5
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(input)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


n = net()
print(n)
# params = list(n.parameters())
# for name, parameters in n.named_parameters():
    # print(name, ":", parameters.size())

input = torch.rand(1, 1,32,32)
out = n(input)
criterion = nn.CrossEntropyLoss()
target = torch.arange(0,10).view(1,10)

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)

optimizer.zero_grad()
output = n(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

