# Author: Charler Qian
# Data: 2018.08.20
# Purpose: Learning for TF by build a VGG Model, using the version 1.4.1

import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import setproctitle

# Name the Process
setproctitle.setproctitle("tf_test_VGG")

# Chose the Device of GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def conv_layer(input,conv_shape,name,params,strides=[1,1,1,1]):
    with tf.variable_scope(name):
        conv_filter = tf.get_variable(
                                    name = 'weights',
                                    shape = conv_shape,
                                    initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                    dtype = tf.float32
                                )
        #bias = tf.Variable(tf.constant(0.0,shape=[conv_shape[-1]],dtype = tf.float32),trainable=True,name='bias')
        bias = tf.get_variable(name = 'bias',shape = conv_shape[-1],initializer=tf.zeros_initializer(dtype=tf.float32))
        output = tf.nn.conv2d(input,conv_filter,strides,padding='SAME')
        output = tf.nn.bias_add(output,bias)
        output = tf.nn.relu(output)
        params += [conv_filter,bias]
    return output

def pooling_layer(input,name,padding='SAME',window_shape=[ 2, 2, 1],strides=[2, 2],pooltype='MAX'):
    output = tf.nn.pool(
                        input,
                        window_shape=window_shape,
                        strides=strides,
                        pooling_type = pooltype,
                        padding='SAME',
                        name=name
                        )
    return output

def fc_layer(input,output_shape,name,params):
    with tf.variable_scope(name):
        shape = int(np.prod(input.get_shape()[1:]))
        input = tf.reshape(input,[-1,shape])
        weight = tf.get_variable(
                                    name = 'weights',
                                    shape = [shape,output_shape],
                                    initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                    dtype = tf.float32
                                )
        bias = tf.get_variable(name = 'bias',shape = output_shape,initializer=tf.ones_initializer(dtype=tf.float32))
        output = tf.nn.bias_add(tf.matmul(input,weight),bias)
        output = tf.nn.relu(output)
        params += [weight,bias]
        return output

def inference_VGG(input):
    params = []
    # Block 1
    conv1_1 = conv_layer(input,[3,3,1,64],'conv1_1',params=params)
    conv1_2 = conv_layer(conv1_1,[3,3,64,64],'conv1_2',params=params)
    maxpool1 = pooling_layer(conv1_2,window_shape=[2, 2],strides=[ 2, 2],pooltype='MAX',name='maxpool1')
    # Block 2
    conv2_1 = conv_layer(maxpool1,[3,3,64,128],'conv2_1',params=params)
    conv2_2 = conv_layer(conv2_1,[3,3,128,128],'conv2_2',params=params)
    maxpool2 = pooling_layer(conv2_2,window_shape=[2, 2],strides=[ 2, 2],pooltype='MAX',name='maxpool2')
    # Block 3
    conv3_1 = conv_layer(maxpool2,[3,3,128,256],'conv3_1',params=params)
    conv3_2 = conv_layer(conv3_1,[3,3,256,256],'conv3_2',params=params)
    conv3_3 = conv_layer(conv3_2,[3,3,256,256],'conv3_3',params=params)
    maxpool3 = pooling_layer(conv3_3,window_shape=[2, 2],strides=[ 2, 2],pooltype='MAX',name='maxpool3')
    # FC layers
    fc1 = fc_layer(maxpool3,4096,'FC1',params=params)
    fc2 = fc_layer(fc1,4096,'FC2',params=params)
    fc3 = fc_layer(fc2,10,'FC3',params=params)
    inference_output = tf.nn.softmax(fc3)
    return inference_output,params,fc3

def loss_vgg(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')

def train(loss):
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

BATCH_SIZE = 32
data = input_data.read_data_sets('fashion-mnist-master/data/fashion')
img_data,label = data.train.next_batch(BATCH_SIZE)
img_data = img_data.reshape([BATCH_SIZE,28,28,1])
sess = tf.Session()
img = tf.placeholder(tf.float32,[None,28,28,1])
inference_output,params,fc3 = inference_VGG(img)
tf.global_variables_initializer()
res_out= sess.run([inference_output,params,fc3],feed_dict={img:img_data})
print(inference_output)
'''
if __name__ == '__main__':
    sess = tf.Session()
    with sess.as_default():
        with tf.device("/device:GPU:1"):
            img = tf.placeholder(tf.float32,[None,28,28])
            label = tf.placeholder(tf.float32,[None])
            inference_output,params,fc3 = inference_VGG(img)
            loss = loss_vgg(inference_output,label)
            print(dir(loss.graph))
            tf.global_variables_initializer().run()
            img_data = Image.open('./demo.jpg')
            img_data = np.asarray(img_data.resize((224,224),Image.ANTIALIAS),dtype=np.float32)
            #img_data = img_data[np.newaxis,:,:,:]
            img_data = np.stack((img_data,img_data),axis=0)
            #img_data = tf.cast(img_data,tf.float32)
            labels_data = np.array([3,5])
            for i in range(100):
                print(i)
                #res_out= sess.run([inference_output,params,fc3],feed_dict={img:img_data})
                #loss_output = sess.run([loss],feed_dict={inference_output:res_out[0],label:labels_data})
                train_op = train(loss)
                sess.run(train_op,feed_dict={img:img_data,label:labels_data})
                #print(loss_output)            '''
        