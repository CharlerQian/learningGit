# Author: Charler Qian
# Data: 2018.08.20
# Purpose: Learning for TF by build a VGG Model, using the version 1.4.1

import tensorflow as tf
from PIL import Image
import numpy as np
def conv_layer(input,conv_shape,name,strides=[1,1,1,1]):
    with tf.variable_scope(name):
        conv_filter = tf.Variable(
                                    tf.truncated_normal(conv_shape,dtype=tf.float32,stddev=0.1),
                                    name = 'weights'
                                )
        bias = tf.Variable(tf.constant(0.0,shape=[conv_shape[-1]],dtype = tf.float32),trainable=True,name='bias')
        output = tf.nn.conv2d(input,conv_filter,strides,padding='SAME')
        output = tf.nn.bias_add(output,bias)
        output = tf.nn.relu(output)
    return output

def pooling_layer(input,name,padding='SAME',window_shape=[ 2, 2, 1],strides=[ 2, 2, 1],pooltype='MAX'):
    output = tf.nn.pool(
                        input,
                        window_shape=window_shape,
                        strides=strides,
                        pooling_type = pooltype,
                        padding='SAME',
                        name=name
                        )
    return output

def fc_layer(input,output_shape,name):
    with tf.variable_scope(name):
        shape = int(np.prod(input.get_shape()[1:]))
        input = tf.reshape(input,[-1,shape])
        print(input.get_shape())
        weight = tf.Variable(tf.truncated_normal(
                                                    [shape,output_shape],
                                                    dtype = tf.float32,
                                                    stddev=0.1
                                                ),
                            name = 'weights'
                            )
        bias = tf.Variable(tf.constant(1.,shape=[output_shape],dtype=tf.float32),trainable=True,name='biases')
        output = tf.nn.bias_add(tf.matmul(input,weight),bias)
        output = tf.nn.relu(output)
        return output

def inference_VGG(input):
    # Block 1
    conv1_1 = conv_layer(input,[3,3,3,64],'conv1_1')
    conv1_2 = conv_layer(conv1_1,[3,3,64,64],'conv1_2')
    maxpool1 = pooling_layer(conv1_2,window_shape=[ 2, 2],strides=[2, 2],pooltype='MAX',name='maxpool1')
    # Block 2
    conv2_1 = conv_layer(maxpool1,[3,3,64,128],'conv2_1')
    conv2_2 = conv_layer(conv2_1,[3,3,128,128],'conv2_2')
    maxpool2 = pooling_layer(conv2_2,window_shape=[2, 2],strides=[ 2, 2],pooltype='MAX',name='maxpool2')
    # Block 3
    conv3_1 = conv_layer(maxpool2,[3,3,128,256],'conv3_1')
    conv3_2 = conv_layer(conv3_1,[3,3,256,256],'conv3_2')
    conv3_3 = conv_layer(conv3_2,[3,3,256,256],'conv3_3')
    maxpool3 = pooling_layer(conv3_3,window_shape=[2, 2],strides=[ 2, 2],pooltype='MAX',name='maxpool3')
    # Block 4
    conv4_1 = conv_layer(maxpool3,[3,3,256,512],'conv4_1')
    conv4_2 = conv_layer(conv4_1,[3,3,512,512],'conv4_2')
    conv4_3 = conv_layer(conv4_2,[3,3,512,512],'conv4_3')
    maxpool4 = pooling_layer(conv4_3,window_shape=[ 2, 2],strides=[2, 2],pooltype='MAX',name='maxpool4')
    # Block 5
    conv5_1 = conv_layer(maxpool4,[3,3,512,512],'conv5_1')
    conv5_2 = conv_layer(conv4_1,[3,3,512,512],'conv5_2')
    conv5_3 = conv_layer(conv4_2,[3,3,512,512],'conv5_3')
    maxpool5 = pooling_layer(conv5_3,window_shape=[2, 2],strides=[2, 2],pooltype='MAX',name='maxpool5')
    #return maxpool5
    fc1 = fc_layer(maxpool5,200,'FC1')
    return fc1
    fc2 = fc_layer(fc1,[4096],'FC2')
    fc3 = fc_layer(maxpool5,[1000],'FC3')
    inference_output = tf.nn.softmax(fc3)
    return inference_output


if __name__ == '__main__':
    sess = tf.Session()
    with sess.as_default():
        img = tf.placeholder(tf.float32,[None,224,224,3])
        inference_output = inference_VGG(img)
        img_data = Image.open('/home/demo.jpg')
        img_data = img_data.resize((224, 224),Image.ANTIALIAS)
        img_data = np.asarray(img_data,dtype = np.float32)
        img_data = img_data[np.newaxis,:,:,:]
       # print(img_data.shape)
        #conv1 = conv_layer(img,[3,3,3,64],'conv1')
        tf.global_variables_initializer().run()
        #res = sess.run([conv1],feed_dict={img:img_data})
        res = sess.run([inference_output],feed_dict={img:img_data})

