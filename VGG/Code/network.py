# -*- coding: utf-8 -*-
# Author Charler Qian
# Data: 2018.10.15
import tensorflow as tf

## Layer definition
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

## Define the inference network architecture
def inference_Mnist(input):
    params = []
    # Block 1
    conv1_1 = conv_layer(input,[3,3,3,64],'conv1_1',params=params)
    conv1_2 = conv_layer(conv1_1,[3,3,64,64],'conv1_2',params=params)
    maxpool1 = pooling_layer(conv1_2,window_shape=[2, 2],strides=[ 2, 2],pooltype='MAX',name='maxpool1')
    # Block 2
    conv2_1 = conv_layer(maxpool1,[3,3,64,128],'conv2_1',params=params)
    conv2_2 = conv_layer(conv2_1,[3,3,128,128],'conv2_2',params=params)
    maxpool2 = pooling_layer(conv2_2,window_shape=[2, 2],strides=[ 2, 2],pooltype='MAX',name='maxpool2')
'''    # Block 3
    conv3_1 = conv_layer(maxpool2,[3,3,128,256],'conv3_1',params=params)
    conv3_2 = conv_layer(conv3_1,[3,3,256,256],'conv3_2',params=params)
    conv3_3 = conv_layer(conv3_2,[3,3,256,256],'conv3_3',params=params)
    maxpool3 = pooling_layer(conv3_3,window_shape=[2, 2],strides=[ 2, 2],pooltype='MAX',name='maxpool3')
    # Block 4
    conv4_1 = conv_layer(maxpool3,[3,3,256,512],'conv4_1',params=params)
    conv4_2 = conv_layer(conv4_1,[3,3,512,512],'conv4_2',params=params)
    conv4_3 = conv_layer(conv4_2,[3,3,512,512],'conv4_3',params=params)
    maxpool4 = pooling_layer(conv4_3,window_shape=[2, 2],strides=[ 2, 2],pooltype='MAX',name='maxpool4')
    # Block 5
    conv5_1 = conv_layer(maxpool4,[3,3,512,512],'conv5_1',params=params)
    conv5_2 = conv_layer(conv5_1,[3,3,512,512],'conv5_2',params=params)
    conv5_3 = conv_layer(conv5_2,[3,3,512,512],'conv5_3',params=params)
    maxpool5 = pooling_layer(conv5_3,window_shape=[2, 2],strides=[ 2, 2],pooltype='MAX',name='maxpool5')'''
    fc1 = fc_layer(maxpool5,4096,'FC1',params=params)
    fc2 = fc_layer(fc1,4096,'FC2',params=params)
    fc3 = fc_layer(fc2,9,'FC3',params=params)
    inference_output = tf.nn.softmax(fc3)
    return inference_output,params,fc3
