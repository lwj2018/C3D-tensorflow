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
import os
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
from skimage.transform import resize
from skimage import io
from scipy import misc
import scipy.io
from guided_backpro import GuideBackPro
#from grad_cam import ClassifyGradCAM
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

def _variable_on_cpu(name, shape):
  #with tf.device('/cpu:%d' % cpu_id):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape)
  return var

def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape)
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('losses', weight_decay)
  return var

def grad_cam(predicted_class, images_placeholder, test_images, conv_layer, fc_layer, nb_classes = 101 ):
  print("Setting gradients to 1 for target class and rest to 0")
  # Conv layer tensor [?,2,7,7,512]
  # [101] 1D tensor with target class index set to 1 and rest to 0
  one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
  signal = tf.multiply(fc_layer, one_hot)
  loss = tf.reduce_mean(signal)

  grads = tf.gradients(loss, conv_layer)[0]
  # Normalizing the gradients
  norm_grads = tf.div(grads, tf.sqrt(tf.reduce_mean(tf.square(grads))) + tf.constant(1e-5))

  output, grads_val = sess.run([conv_layer, norm_grads], feed_dict = {images_placeholder: test_images})
  output = output[0]             # [2,7,7,512]
  grads_val = grads_val[0]           # [2,7,7,512]

  weights = np.mean(grads_val, axis = (0, 1, 2))        #512
  cam = np.zeros(output.shape[0:3], dtype = np.float32)   #[2,7,7]
  
  # Taking a weighted average
  for i,w in enumerate(weights):
    cam += w * output[:,:,:,i]
  
  # Passing through RELU
  cam = np.maximum(cam, 0)
  cam = cam / np.max(cam)

  # Resize to [16,112,112]
  length = cam.shape[0]
  width = cam.shape[1]
  height = cam.shape[2]
  stride = int(16/length)
  new_cam = np.zeros([16,112,112])
  for l in range(length):
    for s in range(stride):
      new_cam[l*stride + s, :, :] = resize(cam[l, :, :],(112,112))
  
  # Converting grayscale to 3-D
  new_cam3 = np.expand_dims(new_cam , axis = 3)
  new_cam3 = np.tile(new_cam3, [1,1,1,3])

  return  new_cam3

def run_test():
  SAVE_PATH = "output/guided_backpro"
  model_name = "/media/storage/liweijie/c3d_models/pbd_fcn_model-1000"
  test_list_file = 'list/predict_test.txt'
  num_test_videos = len(list(open(test_list_file,'r')))
  print("Number of test videos={}".format(num_test_videos))

  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size * gpu_num)
  with tf.variable_scope('var_name') as var_scope:
    weights = {
              'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
              'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
              'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
              'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
              'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
              'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
              'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
              'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
              'w1': _variable_with_weight_decay('w1', [1, 4, 4, 512, 4096], 0.0005),
              'w2': _variable_with_weight_decay('w2', [1, 1, 1, 4096, 4096], 0.0005),
              #'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], 0.0005)
              'out_fcn':_variable_with_weight_decay('wout', [1, 1, 1, 4096, c3d_model.NUM_CLASSES], 0.0005)
              }
    biases = {
              'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
              'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
              'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
              'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
              'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
              'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
              'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
              'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
              'b1': _variable_with_weight_decay('b1', [4096], 0.000),
              'b2': _variable_with_weight_decay('b2', [4096], 0.000),
              #'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.000),
              'out_fcn':_variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], 0.0005)
              }

  batch_size = FLAGS.batch_size
  with tf.device('/cpu:0'):
    out = c3d_model.inference_c3d_full_conv(
                images_placeholder[:,:,:,:,:],
                1,
                FLAGS.batch_size,
                weights,
                biases
                )
  
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
    #all_steps = int((num_test_videos - 1) / (FLAGS.batch_size * gpu_num) + 1)
    all_steps = 6
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
                      
      label = step
      length = 16
      # guided_backpro
      model = GuideBackPro(weights, biases, vis_model = c3d_model)
      back_pro_op = model.get_visualization(images_placeholder)
      guided_backpro = sess.run(back_pro_op, feed_dict = {images_placeholder: test_images})
      guided_backpro_img = guided_backpro[0][0][0]

      print("the shape of guided_backpro_img is:", guided_backpro_img.shape)
      print("max of img is:",np.max(guided_backpro_img))
      guided_backpro_img = guided_backpro_img / np.max(guided_backpro_img)

      make_path = "{}/{}".format(SAVE_PATH, label)
      if not os.path.exists(make_path):
        os.makedirs(make_path)
      for l in range(length):
        misc.imsave("{}/{}/{}.jpg".format(SAVE_PATH,label,l), np.reshape(guided_backpro_img[0,l,:,:,:], [112,112,3]))
      #plt.show()

    print("done")

def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()
