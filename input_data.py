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

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time

total_index = 0

def get_frames_data(filename, start_frame, num_frames_per_clip=16):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  for i in range(start_frame, start_frame + num_frames_per_clip):
    image_name = '/home/liweijie/C3D/C3D-v1.0/examples/c3d_finetuning/'+str(filename) + '/' + '%06d.jpg' % i
    img = Image.open(image_name)
    img_data = np.array(img)
    ret_arr.append(img_data)
  return ret_arr

def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)
  np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = range(len(lines))
    #random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
    total_index = 0
  global total_index
  # prevent from sample_index out of list , restart from the head
  if total_index + batch_size > len(lines):
    total_index = 0 

  for index in video_indices[total_index:]:  
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    start_frame = int(line[1])

    # DEBUG
    if(batch_index == 0):
      print("read clip from: {}  {}".format(dirname, start_frame))

    if(len(line)!=3): print("readed line:"+line)
    tmp_label = line[2]
    if not shuffle:
      print("Loading a video clip from {}...".format(dirname))
    tmp_data = get_frames_data(dirname, start_frame, num_frames_per_clip)
    img_datas = [];
    if(len(tmp_data)!=0):
      for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if(img.width>img.height):
          scale = float(crop_size)/float(img.height)
          img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
          scale = float(crop_size)/float(img.width)
          img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size)/2)
        crop_y = int((img.shape[1] - crop_size)/2)
        #img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
        img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:]
        img_datas.append(img)
      data.append(img_datas)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      total_index = total_index + 1
      read_dirnames.append(dirname)

  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))

  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)

  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len
