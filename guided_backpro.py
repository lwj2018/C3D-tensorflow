#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: guided_backpro.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import c3d_model

@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
    gate_g = tf.cast(grad > 0, "float32")
    gate_y = tf.cast(op.outputs[0] > 0, "float32")
    return grad * gate_g * gate_y

def global_avg_pool(x, name='global_avg_pool'):
    assert x.shape.ndims == 5
    with tf.name_scope(name):
        axis = [1,2,3]    #data_format = 'NLHWC'
        return tf.reduce_mean(x, axis)


class GuideBackPro(object):
    def __init__(self, _weights, _biases, vis_model=None, class_id=None):
        assert vis_model is not None, 'vis_model cannot be None!'
        # assert not class_id is None, 'class_id cannot be None!'

        self._vis_model = vis_model
        if class_id is not None and not isinstance(class_id, list):
            class_id = [class_id]
        self._class_id = class_id
        self._weights = _weights
        self._biases = _biases

    def _create_model(self, image):
        keep_prob = 1
        self._vis_model.inference_c3d_full_conv(image, 1, 1, self._weights, self._biases)
        self.input_im = self._vis_model.layers['input']

        self._out_act = self._vis_model.layers['output']
        self.pre_label = tf.nn.top_k(
            tf.nn.softmax(self._out_act), k=5, sorted=True)

    def _get_activation(self):
        with tf.name_scope('activation'):
            nclass = self._out_act.shape.as_list()[-1]
            act_list = []
            if self._class_id is None:
                class_list = [self.pre_label.indices[0][0]]
                act_list = [tf.reduce_max(self._out_act)]
            else:
                class_list = self._class_id
                for cid in class_list:
                    one_hot = tf.sparse_to_dense([[cid, 0]], [nclass, 1], 1.0)
                    self._out_act = tf.reshape(self._out_act, [1, nclass])
                    class_act = tf.matmul(self._out_act, one_hot)
                    act_list.append(class_act)

            return act_list, tf.convert_to_tensor(class_list)

    def get_visualization(self, image):
        g = tf.get_default_graph()

        with g.gradient_override_map({'Relu': 'GuidedRelu'}):
            try:
                self._create_model(image)
            except ValueError:
                with tf.variable_scope(tf.get_variable_scope()) as scope:
                    scope.reuse_variables()
                    self._create_model(image)
            act_list, class_list = self._get_activation()

            with tf.name_scope('guided_back_pro_map'):
                guided_back_pro_list = []
                for class_act in act_list:
                    guided_back_pro = tf.gradients(
                        class_act, self._vis_model.layers['input'])
                    guided_back_pro_list.append(guided_back_pro)

                self.visual_map = guided_back_pro_list
                self.class_list = class_list
                return guided_back_pro_list, class_list