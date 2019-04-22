import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#  定义网络的超参数
learning_rate = 0.0001
training_iters = 1000
batch_size = 50
display_step = 100

#  定义网络的参数
n_input = 784  # 输入的维度28*28
n_classes = 10  # 标记的维度0-9
dropout = 0.5

# 输入占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


# 定义卷积
def conv2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x, name=name)


# 定义池化
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)


# 规范化操作
def norm(name, l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


# 定义所有的网络参数
weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96])),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256])),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384])),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),
    'wd1': tf.Variable(tf.random_normal([2 * 2 * 256, 1024])),
    'wd2': tf.Variable(tf.random_normal([1024, 1024])),
    'out': tf.Variable(tf.random_normal([1024, 10]))
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96])),
    'bc2': tf.Variable(tf.random_normal([256])),
    'bc3': tf.Variable(tf.random_normal([384])),
    'bc4': tf.Variable(tf.random_normal([384])),
    'bc5': tf.Variable(tf.random_normal([256])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'bd2': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# 定义整个网络
def alex_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
    pool1 = maxpool2d('pool1', conv1, k=2)
    norm1 = norm('norm1', pool1, lsize=4)

    conv2 = conv2d('conv2', norm1, weights['wc2'], biases['bc2'])
    pool2 = maxpool2d('pool2', conv2, k=2)
    norm2 = norm('norm2', pool2, lsize=4)

    conv3 = conv2d('conv3', norm2, weights['wc3'], biases['bc3'])
    pool3 = maxpool2d('pool3', conv3, k=2)
    norm3 = norm('norm3', pool3, lsize=4)

    conv4 = conv2d('conv4', norm3, weights['wc4'], biases['bc4'])
    conv5 = conv2d("conv5", conv4, weights['wc5'], biases['bc5'])  # 接连两层卷积层
    pool5 = maxpool2d('pool5', conv5, k=2)
    norm5 = norm('norm5', pool5, lsize=4)

    fc1 = tf.reshape(norm5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)

    out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
    return out

# def alex_net(x, weights, biases, dropout):
#     x = tf.reshape(x, shape=[-1, 28, 28, 1])
#
#     conv1 = conv2d('conv1', x, weights['wc1'], biases['bc1'])
#     pool1 = maxpool2d('pool1', conv1, k=2)
#
#     conv2 = conv2d('conv2', pool1, weights['wc2'], biases['bc2'])
#     pool2 = maxpool2d('pool2', conv2, k=2)
#
#     conv3 = conv2d('conv3', pool2, weights['wc3'], biases['bc3'])
#     pool3 = maxpool2d('pool3', conv3, k=2)
#
#     conv4 = conv2d('conv4', pool3, weights['wc4'], biases['bc4'])
#     conv5 = conv2d("conv5", conv4, weights['wc5'], biases['bc5'])  # 接连两层卷积层
#     pool5 = maxpool2d('pool5', conv5, k=2)
#
#     # fc1 = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
#     fc1 = tf.reshape(pool5, [-1, 2 * 2 * 256])
#     fc1 = tf.nn.relu(tf.matmul(fc1, weights['wd1'])+biases['bd1'])
#     fc1_drop = tf.nn.dropout(fc1, dropout)
#
#     # fc2 = tf.reshape(fc1, [-1, weights['wd2'].get_shape().as_list()[0]])
#     # fc2 = tf.reshape(fc1, [-1, 4096])
#     fc2 = tf.nn.relu(tf.matmul(fc1_drop, weights['wd2'])+biases['bd2'])
#     fc2_drop = tf.nn.dropout(fc2, dropout)
#
#     out = tf.add(tf.matmul(fc2_drop, weights['out']), biases['out'])
#     return out


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    pred = alex_net(x, weights, biases, keep_prob)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for step in range(1000):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            if step % display_step == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={
                    x: batch_x, y: batch_y, keep_prob: 1.0})
                print("step %d, accuracy %g" % (step, train_accuracy))
                # loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                # print('Iter: %d ' + str(step) + "loss&accuracy" + str(sess.run(tf.cast(loss, tf.float32))) + " & "
                #       + str(sess.run(tf.cast(accuracy, tf.float32))))
        print("test accuracy %g" % accuracy.eval(session=sess, feed_dict={
            x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0
        }))

