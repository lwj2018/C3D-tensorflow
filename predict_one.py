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
from skimage.transform import resize
from skimage import io

from guided_backpro import GuideBackPro

from tensorcv.utils.viz import image_overlay, save_merge_images
from scipy import misc
from guided_backpro import global_avg_pool
from itertools import count
from utils.viz import image_weight_mask

class BaseGradCAM(object):
    def __init__(self, _weights, _biases, vis_model=None, num_channel=3):
        self._vis_model = vis_model
        self._nchannel = num_channel
        self._weights = _weights
        self._biases = _biases

    def create_model(self, inputs):
        self._vis_model.inference_c3d_full_conv(inputs, 1, 1, self._weights, self._biases)

    def _create_model(self, inputs):
        pass

    def setup_graph(self):
        pass

    def _comp_feature_importance_weight(self, class_id):
        if not isinstance(class_id, list):
            class_id = [class_id]

        with tf.name_scope('feature_weight'):
            self._feature_w_list = []
            for idx, cid in enumerate(class_id):
                one_hot = tf.sparse_to_dense(
                    [[cid, 0]], [self._nclass, 1], 1.0)
                out_act = tf.reshape(self._out_act, [1, self._nclass])
                class_act = tf.matmul(out_act, one_hot,
                                      name='class_act_{}'.format(idx))
                feature_grad = tf.gradients(class_act, self._conv_out,
                                            name='grad_{}'.format(idx))    #[ [1,2,7,7,512] ]    
                feature_grad = tf.squeeze(
                    tf.convert_to_tensor(feature_grad), axis=0)
                feature_w = global_avg_pool(
                    feature_grad, name='feature_w_{}'.format(idx))      #[1,512]    
                self._feature_w_list.append(feature_w)

    def get_visualization(self, class_id=None):
        assert class_id is not None, 'class_id cannot be None!'

        with tf.name_scope('grad_cam'):
            self._comp_feature_importance_weight(class_id)
            conv_out = self._conv_out
            conv_shape = conv_out.shape.as_list()
            conv_c = tf.shape(conv_out)[-1]
            conv_l = conv_shape[1]
            conv_h = tf.shape(conv_out)[2]
            conv_w = tf.shape(conv_out)[3]
            #conv_reshape = tf.reshape(conv_out, [conv_l * conv_h * conv_w, conv_c])

            o_l = tf.shape(self.input_im)[1]
            o_h = tf.shape(self.input_im)[2]
            o_w = tf.shape(self.input_im)[3]

            classmap_list = []
            for idx, feature_w in enumerate(self._feature_w_list):
                feature_w = tf.reshape(feature_w, [conv_c, 1])
                classmap_seq = []
                for l in range(conv_l):
                    conv_image = conv_out[0,l,:,:,:]   # get the l-th image in the feature map seq
                    conv_reshape = tf.reshape(conv_image, [conv_h * conv_w, conv_c])
                    classmap = tf.matmul(conv_reshape, feature_w)
                    classmap = tf.reshape(classmap, [-1, conv_h, conv_w, 1])
                    classmap = tf.nn.relu(
                        tf.image.resize_bilinear(classmap, [o_h, o_w]),
                        name='grad_cam_{}'.format(idx))
                    classmap_seq.append(tf.squeeze(classmap)) 
                    # finally the shape of classmap is[o_h, o_w] 
                classmap_list.append(classmap_seq)

            return classmap_list, tf.convert_to_tensor(class_id)


class ClassifyGradCAM(BaseGradCAM):
    def _create_model(self, inputs):
        keep_prob = 1
        self._vis_model.create_model([inputs, keep_prob])

    def setup_graph(self):
        self.input_im = self._vis_model.layers['input']
        self._out_act = self._vis_model.layers['output']
        self._conv_out = self._vis_model.layers['conv_out']
        self._nclass = self._out_act.shape.as_list()[-1]
        self.pre_label = tf.nn.top_k(tf.nn.softmax(self._out_act),
                                     k=5, sorted=True)


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

def grad_cam(predicted_class, images_placeholder, test_images, conv_layer, fc_layer, nb_classes = 6 ):
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

def get_guided_back_pro(predicted_class, images_placeholder, test_images, conv_layer, fc_layer, nb_classes = 6 ):
  print("start get guided back pro")
  # Conv layer tensor [?,2,7,7,512]
  # [101] 1D tensor with target class index set to 1 and rest to 0
  one_hot = tf.sparse_to_dense(predicted_class, [nb_classes], 1.0)
  signal = tf.multiply(fc_layer, one_hot)
  loss = tf.reduce_mean(signal)

  back_pro_input = sess.run(tf.gradients(loss, images_placeholder)[0], feed_dict = {images_placeholder: test_images})
  print('the shape of back_pro_input is: ',back_pro_input.shape)

  back_pro_image = np.reshape(back_pro_input[0,10,:,:,:],(112,112,3))
  plt.imshow(back_pro_image)
  plt.show()

  return back_pro_input

def run_test():
  SAVE_PATH = "output/grad_cam"
  grid_size = 4
  LENGTH = 16
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
  with tf.device('/gpu:0'):
    saver = tf.train.Saver()
    #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    
    init = tf.global_variables_initializer()
    sess.run(init)
    # Create a saver for writing training checkpoints.
    saver.restore(sess, model_name)
    # And then after everything is built, start the training loop.
    bufsize = 0
    next_start_pos = 0

    # -----------------GRAD_CAM(setup)-------------
    print("\033[0;31m start the setup operation \033[0m")
    # create model 
    gcam = ClassifyGradCAM(weights, biases, vis_model = c3d_model)
    gbackprob = GuideBackPro(weights, biases, vis_model = c3d_model)
    gcam.create_model(images_placeholder)
    gcam.setup_graph()
    
    all_steps = 6
    o_im_list = []
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

      
      # -------------------GRAD_CAM(execute)---------------------
      class_id = [ step ]
      # generate class map and prediction label ops
      map_op = gcam.get_visualization(class_id = class_id)
      label_op = gcam.pre_label

      back_pro_op = gbackprob.get_visualization(images_placeholder)
      # execute the ops
      # gcam_map = [ [class_seq], [class_id] ] , in which
      # class_seq is a list of class_map
      # the shape of o_im is NLHWC
      gcam_map, b_map, label, o_im = \
        sess.run([map_op, back_pro_op, label_op, gcam.input_im], feed_dict = {images_placeholder: test_images})
      o_im_list.extend(o_im)
      cnt = 0
      for idx, cid, cmaps in zip(count(), gcam_map[1], gcam_map[0]):
        overlay_im_list = []
        weight_im_list = []
        cmaps_length = len(cmaps)
        stride = int(LENGTH/cmaps_length)
        for l in range(LENGTH):
          cmap = cmaps[int(l/stride)]
          overlay_im = image_overlay(cmap, o_im[:,l,:,:,:])
          weight_im = image_weight_mask(b_map[0][0][0][:,l,:,:,:], cmap)
          overlay_im_list.append(overlay_im)
          weight_im_list.append(weight_im)
        cnt += 1
        save_path = "{}/class_{}_gradcam_{}.png".format(SAVE_PATH,cid, cnt)
        save_merge_images(np.array(overlay_im_list), [grid_size,grid_size], save_path)
        save_path = "{}/class_{}_guided_gradcam_{}.png".format(SAVE_PATH,cid, cnt)
        save_merge_images(np.array(weight_im_list), [grid_size,grid_size], save_path)

      # # grad_cam
      # label = step
      # cam3 = grad_cam(label, images_placeholder, test_images, c3d_model.layers['conv4'], out, nb_classes = 6 )
      # for l in range(cam3.shape[0]):
      #   tempImg = np.reshape(test_images[0,l,:,:,:],(112,112,3))
      #   #print("the minimum of image is : {}".format(tempImg.min()))
      #   tempImg /= np.abs(tempImg).max()
      #   tempImg = tempImg.astype(np.float32)

      #   new_img = tempImg + cam3[l]
      #   new_img /= np.abs(new_img).max()   
      #   # print("the max of cam is : ",np.max(cam3[l]))   

      #   #Dislay and save
      #   io.imsave("output/grad_cam/{}/{:0>6d}.jpg".format(label,l+1), new_img)


    print("done")

def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()
