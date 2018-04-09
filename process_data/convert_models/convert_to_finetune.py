# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Convert classification.caffe2 models to init detection.caffe2 models
#
# tools/pickle_caffe2_blobs.py \
#   --c2_model /path/to/c2/cls/model.pkl \
#   --output /path/to/converted/model.pkl \
#   --absorb_std

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
import numpy as np


def remove_spatial_bn_layers(c2cls_weights):
    finished_bn_blobs = []
    blobs = sorted(c2cls_weights['blobs'].keys())
    for blob in blobs:
        idx = blob.find('_bn_')
        if idx > -1:
            bn_layername = blob[:idx]
            if bn_layername not in finished_bn_blobs:
                finished_bn_blobs.append(bn_layername)

                name_s = bn_layername + '_bn_s'  # gamma
                name_b = bn_layername + '_bn_b'  # beta
                name_rm = bn_layername + '_bn_rm'  # running mean
                name_rv = bn_layername + '_bn_riv'  # running var?

                scale = c2cls_weights['blobs'][name_s]
                bias = c2cls_weights['blobs'][name_b]
                bn_mean = c2cls_weights['blobs'][name_rm]
                bn_var = c2cls_weights['blobs'][name_rv]

                std = np.sqrt(bn_var + 1e-5)
                new_scale = scale / std
                new_bias = bias - bn_mean * scale / std

                # rewrite
                del c2cls_weights['blobs'][name_rm]
                del c2cls_weights['blobs'][name_rv]

                c2cls_weights['blobs'][name_s] = new_scale
                c2cls_weights['blobs'][name_b] = new_bias

    return


def absorb_std_to_conv1(c2cls_weights):
    # ---------------------
    # absorb input std into conv1_w
    # in cls.c2: (raw) img: [0, 255]
    # data = (img / 255 - mean) / std
    # conv1 = data * conv1_w = (img / (255std) - mean / std) * conv1_w
    # -------
    # in det.c2: (raw) img: [0, 255]
    # data = img - (255mean)
    # conv1 = data * conv1_w'
    # ==> conv1_w' = conv1_w / (255std)
    # -------

    DATA_STD = [0.225, 0.225, 0.225]  # B, G, R in cls.c2

    w = c2cls_weights['blobs']['conv1_w']   # (64, 3, 5, 7, 7)
    # assert(w.shape == (64, 3, 7, 7))

    # Note: the input conv1_w to this function is always BGR
    for i in range(3):
        w[:, i, :, :, :] /= (DATA_STD[i] * 255)


def remove_non_param_fields(c2cls_weights):
    fields_to_del = ['epoch', 'model_iter', 'lr']
    for field in fields_to_del:
        if field in c2cls_weights['blobs']:
            del c2cls_weights['blobs'][field]


def remove_momentum(c2cls_weights):
    for k in c2cls_weights['blobs'].keys():
        if k.endswith('_momentum'):
            del c2cls_weights['blobs'][k]


def load_and_convert_caffe2_cls_model(model_file_name):
    with open(model_file_name, 'r') as f:
        c2cls_weights = pickle.load(f)

    # by default, remove lr, epoch, model_iter
    remove_non_param_fields(c2cls_weights)

    # by default, remove momentum
    remove_momentum(c2cls_weights)

    # convert bn into one affine layer
    remove_spatial_bn_layers(c2cls_weights)

    # no need to revert rgb

    # absorb std (input to this should be BGR)
    # absorb_std_to_conv1(c2cls_weights)

    return c2cls_weights
