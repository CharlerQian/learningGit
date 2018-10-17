# -*- coding: utf-8 -*-
# Author Charler Qian
# Data: 2018.10.15

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from network import inference_Mnist
import os
import setproctitle

setproctitle.setproctitle("tf_test_VGG")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

## Firstly Load the data
data = input_data.read_data_sets('data/fashion')


## Then get the inference network architecture
with tf.sess
	inference_Mnist


