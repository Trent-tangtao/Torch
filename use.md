## 一些项目代码中常用的函数

### if __name__ == '__main__'

`if __name__ == '__main__'`的意思是：当.py文件被直接运行时，`if __name__ == '__main__'`之下的代码块将被运行；当.py文件以模块形式被导入时，`if __name__ == '__main__'`之下的代码块不被运行

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

cross entropy 

### 激活函数

sigmoid         ![sigmodå½æ°å¬å¼](https://img-blog.csdn.net/20180104112208199?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![sigmodå½æ°å¾](https://img-blog.csdn.net/20180104111804326?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



tanh             ![tanhå½æ°å¬å¼](https://img-blog.csdn.net/20180104112848849?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



![tanhå½æ°å¾](https://img-blog.csdn.net/20180104113045182?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

relu                ![reluå½æ°å¬å¼](https://img-blog.csdn.net/20180104113836278?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



![reluå½æ°å¾](https://img-blog.csdn.net/20180104114009780?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 过拟合和欠拟合

过拟合：增大数据量； 采用正则化；dropout

欠拟合：增加特征值；构造复杂多项式(泛化)；减少正则化参数