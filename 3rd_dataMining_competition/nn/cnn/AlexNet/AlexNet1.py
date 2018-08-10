
# coding: utf-8

# In[ ]:


import math
import time
import tensorflow as tf
import numpy as np
import cv2
import random
import os


# In[ ]:


BATCH_SIZE = 12
IMAGE_SIZE = 224
NUM_CHANNELS = 3
# LEARNING_RATE_BASE = 0.01
# LEARNING_RATE_DECAY = 0.99
# TRAINING_STEPS = 10000
learning_rate = 0.001
NUM_EXAMPLES = 2726
train_ratio=0.8
MODEL_SAVE_PATH = "brand_model/"
MODEL_NAME = "brand_model"
n_epoch = 100
img_dir = r'D:\Project\jupyter\project\nn\cnn\datasets\train'
text_dir =  r'D:\Project\jupyter\project\nn\cnn\datasets\train.txt' 


# In[ ]:


def conv_op(input_op, name, filter_h, filter_w, out_channels, stride_h, stride_w):
    in_channels = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                shape = [filter_h, filter_w, in_channels, out_channels],
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer_conv2d())
        
        conv = tf.nn.conv2d(input_op, kernel, [1, stride_h, stride_w, 1], padding = 'SAME')
        bias_init = tf.constant(0.0, shape = [out_channels], dtype = tf.float32)
        biases = tf.Variable(bias_init, trainable = True, name = 'b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name = scope)
        
        return activation


# In[ ]:


def fc_op(input_op, name, out_nodes):
    in_nodes = input_op.get_shape()[-1].value
    
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',
                                shape = [in_nodes, out_nodes],
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer_conv2d())
        
        biases = tf.Variable(tf.constant(0.1, shape = [out_nodes], dtype = tf.float32), name = 'b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name = scope)
        
        return activation


# In[ ]:


def pool_op(input_op, name, kh, kw, stride_h, stride_w ):
    pool = tf.nn.max_pool(input_op, 
                          ksize = [1, kh, kw, 1], 
                          strides = [1, stride_h, stride_w, 1], 
                          padding = 'SAME', 
                          name = name)
    
    return pool


# In[ ]:


def inference(input_op, keep_prob = 0.5):
    conv1_1 = conv_op(input_op, name = 'conv1_1', filter_h=3, filter_w=3, out_channels = 64, stride_h=1, stride_w=1)
    conv1_2 = conv_op(conv1_1,  name = 'conv2_2', filter_h=3, filter_w=3, out_channels = 64, stride_h=1, stride_w=1)
    pool1 = pool_op(conv1_2,name='pool1',kh=2,kw=2,stride_h=2,stride_w=2)
    
    conv2_1 = conv_op(pool1, name='conv2_1',   filter_h=3,filter_w=3,out_channels=128,stride_h=1,stride_w=1)
    conv2_2 = conv_op(conv2_1, name='conv2_2', filter_h=3,filter_w=3,out_channels=128,stride_h=1,stride_w=1)
    pool2 = pool_op(conv2_2,name='pool2',kh=2,kw=2,stride_h=2,stride_w=2)
    
    conv3_1 =conv_op(pool2, name='conv3_1',   filter_h=3,filter_w=3,out_channels=256,stride_h=1,stride_w=1)
    conv3_2 =conv_op(conv3_1, name='conv3_2', filter_h=3,filter_w=3,out_channels=256,stride_h=1,stride_w=1)
    conv3_3 =conv_op(conv3_2, name='conv3_3', filter_h=3,filter_w=3,out_channels=256,stride_h=1,stride_w=1)
    pool3 = pool_op(conv3_3,name='pool3',kh=2,kw=2,stride_h=2,stride_w=2)
    
    conv4_1 =conv_op(pool3, name='conv4_1',   filter_h=3,filter_w=3,out_channels=512,stride_h=1,stride_w=1)
    conv4_2 =conv_op(conv4_1, name='conv4_2', filter_h=3,filter_w=3,out_channels=512,stride_h=1,stride_w=1)
    conv4_3 =conv_op(conv4_2, name='conv4_3', filter_h=3,filter_w=3,out_channels=512,stride_h=1,stride_w=1)
    pool4 = pool_op(conv4_3,name='pool4',kh=2,kw=2,stride_h=2,stride_w=2)
    
    conv5_1 =conv_op(pool4, name='conv5_1',   filter_h=3,filter_w=3,out_channels=512,stride_h=1,stride_w=1)
    conv5_2 =conv_op(conv5_1, name='conv5_2', filter_h=3,filter_w=3,out_channels=512,stride_h=1,stride_w=1)
    conv5_3 =conv_op(conv5_2, name='conv5_3', filter_h=3,filter_w=3,out_channels=512,stride_h=1,stride_w=1)
    pool5 = pool_op(conv5_3,name='pool5',kh=2,kw=2,stride_h=2,stride_w=2)
    
    pool5_shape = pool5.get_shape()
    flattened_shape = pool5_shape[1].value * pool5_shape[2].value * pool5_shape[3].value
    pool5_reshape = tf.reshape(pool5, [-1, flattened_shape], name = 'pool5_reshape')
    
    fc6 = fc_op(pool5_reshape, 'fc6', out_nodes = 4096)
    if train:
        fc6 = tf.nn.dropout(fc6, keep_prob, name = 'fc6_dropout')
    
    fc7 = fc_op(fc6, name = 'fc7', out_nodes = 4096)
    if train:
        fc7 = tf.nn.dropout(fc7, keep_prob, name = 'fc7_dropout')
    
    fc8 = fc_op(fc7, name = 'fc8', out_nodes = 100)
    
    return fc8


# In[ ]:


#单个文件夹
def read_img(path, textpath):
    #path = r'D:\Project\jupyter\project\nn\cnn\datasets\train'
    #textpath = r'D:\Project\jupyter\project\nn\cnn\datasets\train.txt'
    dirdata = []
    with open(textpath) as f:
        line = f.readline()
        #i = 0
        while line:
            linelist = line.split()
            linelist[0] = path + '\\' + linelist[0]
            dirdata.append(linelist)
            line = f.readline()
            #i = i+1 
    #print(dirdata)
    imgs=[]
    labels=[]
    print(len(dirdata))
    for data in dirdata:
        #print('reading the images:%s'%(data[0]))
        img=cv2.imread(data[0])
        #img=cv2.resize(img,(w,h))
        img = cv2.resize(img,(224,224),cv2.INTER_LINEAR)
        imgs.append(img)
        labels.append(data[1])
    print("number of examples : ", len(imgs))    
    
#     data = np.asarray(imgs, np.float32)
#     label = np.asarray(labels, np.float32)
    data = np.asarray(imgs, np.float32)
#     print('type data', type(data))
#     print('type(data[0])', type(data[0]))
#     print('type labels[0]:', type(labels[0]))
#     print('type(label):', type(label))
#     print('type label[0]', type(label[0]))
    #label = np.asarray(labels, np.int32)
    label = [int(i) for i in labels] 
    tf_label_onehot = tf.one_hot(label,100)  #需要label为int
    #print(tf_label_onehot)
    with tf.Session() as sess:  
        label = sess.run(tf_label_onehot)
    #return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
     
    return data, label

def get_train_val_data(ratio=0.8):
    data,label=read_img(img_dir, text_dir)


    #打乱顺序
    num_example=data.shape[0]
    #print(num_example)
    arr=np.arange(num_example)
    np.random.shuffle(arr)
    data=data[arr]
    label=label[arr]

    #将所有数据分为训练集和验证集
    #ratio=0.8
    s=np.int(num_example*ratio)
    #print(type(num_example*ratio))
    #print(type(s))
    x_train=data[:s]
    y_train=label[:s]
    x_val=data[s:]
    y_val=label[s:]
    
    return x_train, y_train, x_val, y_val


# In[ ]:


#定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


# In[ ]:


def train():
    x = tf.placeholder(tf.float32, [
        None,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS], name = 'x_input')
    y_ = tf.placeholder(tf.float32, [None, 100], name = 'y_input')
    
    y = inference(x, 0.5)
    #global_step = tf.Variable(0, trainable=False)
    
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    
    loss = cross_entropy_mean
#     learning_rate = tf.train.exponential_decay(
#         LEARNING_RATE_BASE,
#         global_step,
#         NUM_EXAMPLES / BATCH_SIZE,
#         LEARNING_RATE_DECAY,
#         staircase = True
#     )
    
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.control_dependencies([train_step]):  
        train_op = tf.no_op(name = 'train')
    
    train_x, train_y, _, _ = get_train_val_data(0.8)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(n_epoch):
            start_time = time.time()
            
            for xs, ys in minibatches(train_x, train_y, BATCH_SIZE, True):
                _ = sess.run([train_op], feed_dict = {x: xs, y_: ys})
                loss_value = sess.run(loss, feed_dict = {x:xs, y_:ys})
                print("After %d training epoch(s), loss on training batch is %g." %(epoch, loss_value))
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=epoch)   #  注意这里！！！
#         for i in range(TRAINING_STEPS):
#             xs, ys = next_batch(i, data_x, data_y)
#             _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x: xs, y_:ys})
            
#             if i % 5 == 0:
#                 print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
#                 saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


# In[ ]:


def main():
    train()
if __name__ == '__main__':
    main()

