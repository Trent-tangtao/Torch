import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def filter_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    import os
    os.environ['Tf_CPP-MIN-LOG-LEVEL'] = '2'

    x = tf.placeholder('float', shape=[None, 784])
    # 输入图像是由2维的浮点数tensor组成的，这里我们分配给它的形状是[None,784]，
    # 其中784表示一个28x28像素点的MNIST图像单一展开的维度，
    # None表示第一个维度，与batch的大小有关，它可以是任意大小，即输入图像数量不唯一
    y = tf.placeholder('float', shape=[None, 10])

    f_conv1 = filter_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])  # -1表示取出所有的数据

    h_conv1 = tf.nn.relu(conv2d(x_image, f_conv1)+b_conv1)
    h_pool1 = max_pool_2_2(h_conv1)

    f_conv2 = filter_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, f_conv2)+b_conv2)
    h_pool2 = max_pool_2_2(h_conv2)

    # 28 / 2 / 2
    w_fc1 = filter_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = filter_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2)+b_fc2)

    cross_entropy = -tf.reduce_sum(y*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correction_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_conv, 1))
    accuracy = tf.reduce_mean(tf.cast(correction_prediction, 'float'))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={
                    x: batch[0], y: batch[1], keep_prob: 1.0})
                print("step %d, accuracy %g" % (i, train_accuracy))
            train_step.run(session=sess, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
        print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={
            x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0
        }))
        # 打印出测试集的结果
