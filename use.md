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