
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os


# In[2]:


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 9000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "MNIST_model/"
MODEL_NAME = "mnist_model"

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10


# In[3]:


def conv_op(input_op, name, kh, kw, n_out,stride):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
#         kernel = tf.get_variable(scope + 'w',
#                                 shape = [kh, kw, n_in, n_out],
#                                 dtype = tf.float32,
#                                 initializer = tf.truncated_normal_initializer(stddev = 0.1))
        kernel = tf.get_variable(scope + 'w',
                                shape = [kh, kw, n_in, n_out],
                                dtype = tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
        
        conv = tf.nn.conv2d(input_op, kernel, [1, stride, stride, 1], padding = 'SAME')
        bias = tf.get_variable(scope + 'b', 
                              shape = [n_out],
                              dtype = tf.float32,
                              initializer = tf.truncated_normal_initializer(stddev = 0.0))
        z = tf.nn.bias_add(conv, bias)
        activation = tf.nn.relu(z, name = scope)
        
        return activation


# In[4]:


def fc_op(input_op, name, n_out, regularizer):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
#         kernel = tf.get_variable(scope + 'w',
#                                 shape = [n_in, n_out],
#                                 dtype = tf.float32,
#                                 initializer = tf.truncated_normal_initializer(stddev = 0.1))
        kernel = tf.get_variable(scope + 'w',
                                shape = [n_in, n_out],
                                dtype = tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(kernel))
        bias = tf.get_variable(scope + 'b',
                              shape = [n_out],
                              dtype = tf.float32,
                              initializer = tf.truncated_normal_initializer(stddev = 0.0))
        activation = tf.nn.relu_layer(input_op, kernel, bias, name = scope)
        
        return activation


# In[5]:


def pool_op(input_op, name, kh, kw, stride):
    pool = tf.nn.max_pool(input_op, [1,kh, kw, 1], [1, stride, stride, 1], padding = 'SAME', name = name)
    
    return pool


# In[6]:


def inference(input_op, regularizer, train):
    conv1 = conv_op(input_op, 'conv1', 5, 5, 32, 1)
    pool1 = pool_op(conv1, 'pool1', 2, 2, 2)
    conv2 = conv_op(pool1, 'conv2', 5, 5, 64, 1)
    pool2 = pool_op(conv2, 'pool2', 2, 2, 2)
    pool2_shape = pool2.get_shape()
    flattened_shape = pool2_shape[1]*pool2_shape[2]*pool2_shape[3]
    pool2_reshape = tf.reshape(pool2, [-1, flattened_shape], name = 'pool2_reshape')
    
    fc1 = fc_op(pool2_reshape, 'fc1', 512, regularizer)
    if train:
        fc1 = tf.nn.dropout(fc1, 0.5, name = 'fc1_dropout') 
    fc2 = fc_op(fc1, 'fc2', 10, regularizer)
    
    return fc2


# In[ ]:


def train(mnist):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS],
        name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    
    #regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #y = inference(x,False,regularizer)
    y = inference(x, regularizer, True)
    global_step = tf.Variable(0, trainable=False)
   
    #定义滑动平均操作
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 定义损失函数、学习率、滑动平均操作以及训练过程。
#     variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#     variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    #with tf.control_dependencies([train_step, variables_averages_op]):
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
        
    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


# In[ ]:


def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    main()

