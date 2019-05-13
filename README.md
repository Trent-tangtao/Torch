## 一些项目代码中常用的函数

### if __name__ == '__main__'

`if __name__ == '__main__'`的意思是：当.py文件被直接运行时，`if __name__ == '__main__'`之下的代码块将被运行；当.py文件以模块形式被导入时，`if __name__ == '__main__'`之下的代码块不被运行



## Tensorflow

### tf.truncated_normal

tf.truncated_normal(shape, mean, stddev) : shape表示生成张量的维度，mean是均值，stddev是标准差。这个函数产生正太分布，均值和标准差自己设定。这是一个**截断的产生正太分布**的函数，就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。

举例，当输入参数mean = 0 ， stddev =1时，使用tf.truncated_normal的输出是不可能出现[-2,2]以外的点的，而如果shape够大的话，tf.random_normal却会产生2.2或者2.4之类的输出。

### tf.argmax

tf.argmax(vector, 1)：返回的是vector中的**最大值的索引号**，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。

### tf.equal

tf.equal(A, B)是对比这两个矩阵或者向量的**相等**的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的

### tf.Tensor.eval()

当默认的会话被指定之后可以通过其计算一个张量的取值

### tf.app.flags

用于支持接受命令行传递参数，相当于接受argv

第一个是参数名称，第二个参数是默认值，第三个是参数描述

```python
tf.app.flags.DEFINE_string('str_name', 'def_v_1',"descrip1")
tf.app.flags.DEFINE_integer('int_name', 10,"descript2")
tf.app.flags.DEFINE_boolean('bool_name', False, "descript3")

FLAGS = tf.app.flags.FLAGS
```

### tf.cast

cast(x, dtype, name=None)  将x的数据格式转化成dtype

### tf.placeholder()

tf.placeholder() 和 feed_dict  定义变量

tf.constant(‘Hello World!’)  定义常量

```python
x = tf.placeholder(tf.string)
with tf.Session() as sess:
output = sess.run(x, feed_dict={x: 'Hello World'})
```

### One-hot

类别用one hot进行编码，比如共有5个类，那么就有5个编码

[1 0 0 0 0] ,[0 1 0 0 0], [0 0 1 1 1], [0 0 0 1 0] ,[0 0 0 0 1]



### Tf.nn & Tf.train

```python
# 卷积函数
tf.nn.convolution()
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, date_format=None,name=None)
tf.nn.depthwise_conv2d()
tf.nn.separable_conv2d()   #利用几个分离的卷积核
tf.nn.atrous_conv2d()      #孔卷积 or 扩张卷积
tf.nn.conv2d_transpose()    # conv2d 的转置
tf.nn.conv1d()
tf.nn.conv3d()
tf.nn.conv3d_transpose()
```

```python
# 池化
tf.nn.avg_pool()
tf.nn.max_pool()
tf.nn.max_pool_with_argmax()     # 并且算出位置argmax，  只能在GPU下计算
tf.nn.avg_pool3d()
tf.nn.max_pool3d()
tf.nn.fractional_avg_pool()     # 三维
tf.nn.fractional_max_pool()   
tf.nn.pool()
```

```python
# 分类函数
tf.nn.sigmoid_cross_entropy_with_logits()  #内部进行了sigmoid,网络的最后一层不需要sigmoid
tf.nn.softmax()
tf.nn.log_softmax()
tf.nn.softmax_cross_entropy_with_logits()
tf.nn.sparse_softmax_cross_entropy_with_logits()
```

```python
# 优化算法
tf.train.GradientDescentOptimizer()
tf.train.AdadeltaOptimizer()
tf.train.AdagradOptimizer()
tf.train.AdagradDAOOptimizer()
tf.train.MomentumOptimizer()
tf.train.AdamOptimizer()
tf.train.FtrlOptimizer()
tf.train.RMSPropOptimizer()

# BGD 批梯度下降
# SGD 随机梯度下降
```





## Pytorch

### nn.ReLU与F.ReLU

```python
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

class AlexNet_1(nn.Module):

def __init__(self, num_classes=n):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
     )

def forward(self, x):
    x = self.features(x)

class AlexNet_2(nn.Module):

def __init__(self, num_classes=n):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
     )

def forward(self, x):
    x = self.features(x)
    x = F.ReLU(x)
```

在如上网络中，AlexNet_1与AlexNet_2实现的结果是一致的，但是可以看到将ReLU层添加到网络有两种不同的实现，即nn.ReLU和F.ReLU两种实现方法。

其中**nn.ReLU作为一个层结构**，必须添加到nn.Module容器中才能使用，而**F.ReLU则作为一个函数调用**，看上去作为一个函数调用更方便更简洁。具体使用哪种方式，取决于编程风格。在PyTorch中,nn.X都有对应的函数版本F.X，但是并不是所有的F.X均可以用于forward或其它代码段中，因为当网络模型训练完毕时，在存储model时，在**forward中的F.X函数中的参数是无法保存的**。也就是说，**在forward中，使用的F.X函数一般均没有状态参数**，比如F.ReLU，F.avg_pool2d等，均没有参数，它们可以用在任何代码片段中。







# ML/DL常见的概念

### 偏差bias和方差variance

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20180905202253853?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbmF0XzMwMzUzMjU5/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

偏差（拟合不够）指的是算法的期望预测与真实预测之间的偏差程度， 反应了模型本身的拟合能力； 

方差（过度拟合）度量了同等大小的训练集的变动导致学习性能的变化，刻画了数据扰动所导致的影响

### 正则化regularization

**抑制过拟合**   常用的正则化方法是在损失函数(Cost Function)中添加一个系数的l1−normm或l2−norm项，用来抑制过大的模型参数，从而缓解过拟合现象。

**L2正则化 **  通过对大权重增加惩罚项以降低模型复杂度的一种方法

**L1正则化**   通过对大权重增加惩罚项以降低模型复杂度的一种方法

### 梯度下降

![img](https://images2015.cnblogs.com/blog/743682/201511/743682-20151108172551399-1795553319.png)



**各种梯度下降详情见链接**：

 [**批量梯度下降(BGD)、随机梯度下降(SGD)以及小批量梯度下降(MBGD)的理解**](https://www.cnblogs.com/lliuye/p/9451903.html)



### 精确率和召回率

二分类：TP(真阳性) FP(假阳性) TN(真阴性) FN(假阴性)

**精确率precision= TP/(TP+FP)   正确预测为正占全部预测为正的比例**

**召回率recall=TP/(TP+FN)   正确预测为正占全部正样本的比例**

**准确率accuracy  = (TN+TP)/(TP+TN+FN+FP)**

### Softmax

softmax是一个全连接层，功能是将卷积神经网络计算后的多个神经元输出，映射到（0，1）区间，给出每种分类的概率情况

![image](https://images2015.cnblogs.com/blog/961754/201612/961754-20161204172019240-1507380126.png)

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20180902220822202?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JpdGNhcm1hbmxlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



**cross entropy** 

![img](https://img-blog.csdnimg.cn/20190206223310433.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2MzY4Mzg4,size_16,color_FFFFFF,t_70)

### 激活函数

sigmoid         ![sigmodå½æ°å¬å¼](https://img-blog.csdn.net/20180104112208199?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![sigmodå½æ°å¾](https://img-blog.csdn.net/20180104111804326?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20180415160228709?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3R5aGpfc2Y=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



tanh             ![tanhå½æ°å¬å¼](https://img-blog.csdn.net/20180104112848849?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



![tanh(x)åå¶å¯¼æ°çå ä½å¾å](https://img-blog.csdn.net/2018041517590341?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3R5aGpfc2Y=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

relu                ![reluå½æ°å¬å¼](https://img-blog.csdn.net/20180104113836278?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20180503231727530?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3R5aGpfc2Y=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

### 过拟合和欠拟合

过拟合：增大数据量； 采用正则化；dropout

欠拟合：增加特征值；构造复杂多项式(泛化)；减少正则化参数