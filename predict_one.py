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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import numpy as np
#import math

# Basic model parameters as external flags.
flags = tf.app.flags
gpu_num = 1
flags.DEFINE_integer('batch_size', 1 , 'Batch size.')
FLAGS = flags.FLAGS

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

def conv3d(name, l_input, w, b):
  return tf.nn.bias_add(
          tf.nn.conv3d(l_input, w, strides=[1, 1, 1, 1, 1], padding='SAME'),
          b
          )

def conv3d_transpose(name, conv, w, b, shape):
  temp = tf.nn.bias_add(conv, -b)
  return tf.nn.conv3d_transpose(temp, w, shape, strides = [1,1,1,1,1], padding = 'SAME')

def max_pool(name, l_input, k):
  return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def avg_pool(name, l_input, k):
  return tf.nn.avg_pool3d(l_input, ksize=[1, k, 2, 2, 1], strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def un_avg_pool(name, l_input, k, images_placeholder, test_images):
  """
    the input is an 5-D array: batch_size length width height channels
  """
  input_shape = l_input.get_shape().as_list()
  batch_size = input_shape[0]
  length = input_shape[1]
  width = input_shape[2]
  height = input_shape[3]
  channels = input_shape[4]
  output_shape = [batch_size,k*length,2*width,2*height,channels]

  sess.run(tf.global_variables_initializer())
  input_array = np.zeros(input_shape,dtype = np.float32)
  output_array = np.zeros(output_shape,dtype = np.float32)
  input_array = l_input.eval(session = sess,feed_dict={images_placeholder:test_images})
  #input_array = np.array(l_input.as_list())

  for n in range(batch_size):
    for l in range(k*length):
      for w in range(2*width):
        for h in range(2*height):
          for c in range(channels):
            output_array[n,l,w,h,c] = input_array[n,int(l/k),int(w/2),int(h/2),c]
  output = tf.convert_to_tensor(output_array)
  return output

def un_max_pool(name, l_input, l_output, k, images_placeholder, test_images):
  '''
  parameters:
    l_input is the input of pool
    l_output is the output of pool
    according to input,we can get the max index
    according to output and max_index, unpool and reconstruct the input 
  return:
    the reconstructed input
  '''
  input_shape = l_input.get_shape().as_list()
  output_shape = l_output.get_shape().as_list()
  batch_size = output_shape[0]
  length = output_shape[1]
  rows = output_shape[2]
  cols = output_shape[3]
  channels = output_shape[4]
  input_array = l_input.eval(session = sess,feed_dict={images_placeholder:test_images})
  output_array = l_output.eval(session = sess,feed_dict = {images_placeholder:test_images})
  unpool_array = np.zeros(input_shape,dtype = np.float32)
  for n in range(batch_size):
    for l in range(length):
      for r in range(rows):
        for c in range(cols):
          for ch in range(channels):
            l_in, r_in, c_in = k*l, 2*r, 2*c
            sub_square = input_array[ n, l_in:l_in+k, r_in:r_in+2, c_in:c_in+2, ch ]
            max_pos_l, max_pos_r, max_pos_c = np.unravel_index(np.nanargmax(sub_square), (k, 2, 2))
            array_pixel = output_array[ n, l, r, c, ch ]
            unpool_array[n, l_in + max_pos_l, r_in + max_pos_r, c_in + max_pos_c, ch] = array_pixel
  unpool = tf.convert_to_tensor(unpool_array)

  return unpool





def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd):
  var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

def run_test():
  model_name = "./models/c3d_ucf_model-200"
  test_list_file = 'list/predict_test.txt'
  num_test_videos = len(list(open(test_list_file,'r')))
  print("Number of test videos={}".format(num_test_videos))

  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
  with tf.variable_scope('var_name') as var_scope:
    weights = {
            'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.04, 0.00),
            'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.04, 0.00),
            'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.04, 0.00),
            'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.04, 0.00),
            'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.04, 0.00),
            'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.04, 0.00),
            'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.04, 0.001),
            'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.04, 0.002),
            'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.04, 0.005)
            }
    biases = {
            'bc1': _variable_with_weight_decay('bc1', [64], 0.04, 0.0),
            'bc2': _variable_with_weight_decay('bc2', [128], 0.04, 0.0),
            'bc3a': _variable_with_weight_decay('bc3a', [256], 0.04, 0.0),
            'bc3b': _variable_with_weight_decay('bc3b', [256], 0.04, 0.0),
            'bc4a': _variable_with_weight_decay('bc4a', [512], 0.04, 0.0),
            'bc4b': _variable_with_weight_decay('bc4b', [512], 0.04, 0.0),
            'bc5a': _variable_with_weight_decay('bc5a', [512], 0.04, 0.0),
            'bc5b': _variable_with_weight_decay('bc5b', [512], 0.04, 0.0),
            'bd1': _variable_with_weight_decay('bd1', [4096], 0.04, 0.0),
            'bd2': _variable_with_weight_decay('bd2', [4096], 0.04, 0.0),
            'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.04, 0.0),
            }
  _weights = weights
  _biases = biases
  _X = images_placeholder[0: FLAGS.batch_size,:,:,:,:]
  _dropout = 0.6
  batch_size = FLAGS.batch_size
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
  out = tf.matmul(dense2, _weights['out']) + _biases['out']
  
  logits = []
      
  logits.append(out)
  logits = tf.concat(logits,0)
  norm_score = tf.nn.softmax(logits)
  saver = tf.train.Saver()
  #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  saver.restore(sess, model_name)
  # And then after everything is built, start the training loop.
  bufsize = 0
  next_start_pos = 0
  all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
  for step in xrange(all_steps):
    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    start_time = time.time()
    test_images, test_labels, next_start_pos, _, valid_len = \
            input_data.read_clip_and_label(
                    test_list_file,
                    FLAGS.batch_size * gpu_num,
                    start_pos=next_start_pos
                    )

    # # Deconvolution Layer
    # #deconv5 = tf.nn.relu(conv5, 'derelu5b')
    # deconv5 = conv3d_transpose('deconv5b', conv5,_weights['wc5b'],_biases['bc5b'], [1,2,7,7,512])
    # deconv5 = tf.nn.relu(deconv5, 'derelu5a')
    # depool4 = conv3d_transpose('deconv5a', deconv5, _weights['wc5a'], _biases['bc5a'], [1,2,7,7,512])

    # # Deconvolution Layer
    # deconv4 = un_avg_pool('depool4', depool4, k=2, images_placeholder = images_placeholder, test_images = test_images)
    # deconv4 = tf.nn.relu(deconv4, 'derelu4b')
    # deconv4 = conv3d_transpose('deconv4b', deconv4, _weights['wc4b'], _biases['bc4b'], [1,4,14,14,512])
    # deconv4 = tf.nn.relu(deconv4, 'derelu4a')
    # depool3 = conv3d_transpose('deconv4a', deconv4, _weights['wc4a'], _biases['bc4a'], [1,4,14,14,256])

    # # Deconvolution Layer
    # deconv3 = un_avg_pool('depool3', depool3, k=2, images_placeholder = images_placeholder, test_images = test_images)
    # deconv3 = tf.nn.relu(deconv3, 'derelu3b')
    # deconv3 = conv3d_transpose('deconv3b', deconv3, _weights['wc3b'], _biases['bc3b'], [1,8,28,28,256])
    # deconv3 = tf.nn.relu(deconv3, 'derelu3a')
    # depool2 = conv3d_transpose('deconv3a', deconv3, _weights['wc3a'], _biases['bc3a'], [1,8,28,28,128])

    # # Deconvolution Layer
    # deconv2 = un_avg_pool('depool2', pool2, k=2, images_placeholder = images_placeholder, test_images = test_images)
    deconv2 = tf.nn.relu(conv2, 'derelu2')
    depool1 = conv3d_transpose('deconv2', deconv2, _weights['wc2'], _biases['bc2'], [1,16,56,56,64])

    # Deconvolution Layer
    deconv1 = un_max_pool('depool1', conv1, depool1, k=1, images_placeholder = images_placeholder, test_images = test_images)
    deconv1 = tf.nn.relu(deconv1, 'derelu1')
    deconv1 = conv3d_transpose('deconv1', deconv1, _weights['wc1'], _biases['bc1'], [1,16,112,112,3])

    #print(conv5.get_shape())
    # shape of pool3:1 4 14 14 256
    pool3_show = deconv1.eval(session = sess,feed_dict={images_placeholder: test_images})
    print("the shape of pool3 is:{}".format(pool3_show.shape))    
    tempMat = np.reshape(pool3_show[:,10,:,:,:], (112,112,3))
    plt.imshow(tempMat)
    plt.show()
    
    # save images get by pool3
    # #print(conv5.get_shape())
    # # shape of pool3:1 4 14 14 256
    # pool3_show = pool3.eval(session = sess,feed_dict={images_placeholder: test_images})
    # print("the shape of pool3 is:{}".format(pool3_show.shape))    
    # for ch in range(pool3_show.shape[4]):
    #   tempMat = np.asmatrix(pool3_show[:,1,:,:,ch])
    #   plt.imsave("output/pool3/{:0>6d}.jpg".format(ch), tempMat)

    predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: test_images}
            )
    
    for i in range(0, valid_len):
      true_label = test_labels[i],
      top1_predicted_label = np.argmax(predict_score[i])
    print("the predicted label is: %d"%top1_predicted_label)
    
  print("done")

def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()
