# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the C3D network.

Implements the inference pattern for model building.
inference_c3d(): Builds the model as far as is required for running the network
forward to make predictions.
"""

import tensorflow as tf
from guided_backpro import global_avg_pool

# The UCF-101 dataset has 101 classes
NUM_CLASSES = 6

# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
CHANNELS = 3

# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 16

# the dictionary of layers
layers = {}

"-----------------------------------------------------------------------------------------------------------------------"

def conv3d(name, l_input, w, b):
  return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b
          )

def conv3d_transpose(name, conv, w, b, shape):
  return tf.nn.bias_add(
          tf.nn.conv3d_transpose(conv, w, shape, strides = [1,1,1,1,1], padding = 'SAME'),
          -b
          )

def max_pool(name, l_input, k):
  return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def avg_pool(name, l_input, k):
  return tf.nn.avg_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def inference_c3d(_X, _dropout, batch_size, _weights, _biases):

  with tf.device('/gpu:0'):
    # Convolution Layer
    conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = max_pool('pool1', conv1, k=1)

    # Convolution Layer
    conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    pool2 = max_pool('pool2', conv2, k=2)

    # Convolution Layer
    conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    pool3 = max_pool('pool3', conv3, k=2)

    # Convolution Layer
    conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    pool4 = max_pool('pool4', conv4, k=2)

    # Convolution Layer
    conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')
    pool5 = max_pool('pool5', conv5, k=2)

    # Fully connected layer
    pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
    dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv3 output to fit dense layer input
    dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

    dense1 = tf.nn.relu(dense1, name='fc1') # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout)

    dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
    dense2 = tf.nn.dropout(dense2, _dropout)

    # Output: class prediction
    out = tf.matmul(dense2, _weights['out_changed']) + _biases['out_changed']

    global layers
    layers['input'] = _X
    layers['conv1'] = conv1
    layers['conv2'] = conv2
    layers['conv3'] = conv3
    layers['conv4'] = conv4
    layers['conv_out'] = conv4
    layers['pool5'] = pool5
    layers['output'] = out

  return out

def inference_c3d_full_conv(_X, _dropout, batch_size, _weights, _biases):
  with tf.device('/gpu:0'):
    # Convolution Layer
    conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
    conv1 = tf.nn.relu(conv1, 'relu1')
    pool1 = max_pool('pool1', conv1, k=1)

    # Convolution Layer
    conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
    conv2 = tf.nn.relu(conv2, 'relu2')
    pool2 = max_pool('pool2', conv2, k=2)

    # Convolution Layer
    conv3 = conv3d('conv3a', pool2, _weights['wc3a'], _biases['bc3a'])
    conv3 = tf.nn.relu(conv3, 'relu3a')
    conv3 = conv3d('conv3b', conv3, _weights['wc3b'], _biases['bc3b'])
    conv3 = tf.nn.relu(conv3, 'relu3b')
    pool3 = max_pool('pool3', conv3, k=2)

    # Convolution Layer
    conv4 = conv3d('conv4a', pool3, _weights['wc4a'], _biases['bc4a'])
    conv4 = tf.nn.relu(conv4, 'relu4a')
    conv4 = conv3d('conv4b', conv4, _weights['wc4b'], _biases['bc4b'])
    conv4 = tf.nn.relu(conv4, 'relu4b')
    pool4 = max_pool('pool4', conv4, k=2)

    # Convolution Layer
    conv5 = conv3d('conv5a', pool4, _weights['wc5a'], _biases['bc5a'])
    conv5 = tf.nn.relu(conv5, 'relu5a')
    conv5 = conv3d('conv5b', conv5, _weights['wc5b'], _biases['bc5b'])
    conv5 = tf.nn.relu(conv5, 'relu5b')      #[1,4,14,14,512]
    pool5 = max_pool('pool5', conv5, k=2)

    # Fully connected layer
    #pool5 = tf.transpose(pool5, perm=[0,1,4,2,3])
    full_conv1 = conv3d('fcn1', pool5, _weights['w1'], _biases['b1'])
    full_conv2 = conv3d('fcn2', full_conv1, _weights['w2'], _biases['b2'])
    full_conv3 = conv3d('fcn3', full_conv2, _weights['out_fcn'], _biases['out_fcn'])

    # Output: class prediction
    out = global_avg_pool(full_conv3)   #[1,6]


    global layers
    layers['input'] = _X
    layers['conv1'] = conv1
    layers['conv2'] = conv2
    layers['conv3'] = conv3
    layers['conv4'] = conv4
    layers['conv_out'] = conv4
    layers['pool5'] = pool5
    layers['fcn1'] = full_conv1
    layers['fcn2'] = full_conv2
    layers['fcn3'] = full_conv3
    layers['output'] = out

  return out
