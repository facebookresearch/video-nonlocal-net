# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
##############################################################################
#
# Based on:
# Copyright (c) 2017-present, Facebook, Inc.
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
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
This script contains some helpful functions to visualize the net and data
"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

from visdom import Visdom
import math
import os
import numpy as np
import logging

import dataset.image_processor as imgproc

viz = Visdom()
logger = logging.getLogger(__name__)


def visualize_net(model, path, minimal=False):
    from caffe2.python import net_drawer
    net_name = model.net.Proto().name
    if minimal:
        graph = net_drawer.GetPydotGraphMinimal(
            model.net.Proto().op, net_name,
            minimal_dependency=True
        )
    else:
        graph = net_drawer.GetPydotGraph(
            model.net.Proto().op, net_name,
        )
    if minimal:
        img_output_path = os.path.join(path, net_name + '_minimal.png')
    else:
        img_output_path = os.path.join(path, net_name + '_full.png')
    with open(img_output_path, 'w') as fopen:
        fopen.write(graph.create_png())
        logger.info("{}: Net image saved to: {}".format(net_name, img_output_path))


# minibatch visualization using Visdom
def visdom_mini_batch(data):
    # TODO: the data might be processed data and not original images
    # data
    for idx in range(data.shape[0]):
        img = data[idx, :, :, :]
        viz.image(
            img,
            opts=dict(title='Minibatch image {}'.format(idx)),
        )


# using matplotlib to plot images, it expects the data to be float and
# to be in range(0, 1)
def normalize_image_scale(image):
    image = image - np.min(image)
    image /= np.max(image) + np.finfo(np.float64).eps
    return image


def visualize_image(
    image, path, name, order='CHW', normalize=True, channel_order='BGR'
):
    import matplotlib.pyplot as plt
    # if image is in CHW format. Change it to HWC for plotting and RBG order
    if order == 'CHW':
        image = imgproc.CHW2HWC(image)
    if channel_order == 'BGR':
        image = image[:, :, [2, 1, 0]]
    if normalize:
        image = normalize_image_scale(image)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(image, aspect='equal')
    plt.axis('off')
    plt.tight_layout()
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(
        os.path.join(path, name), bbox_inches='tight', dpi='figure',
        transparent="True", pad_inches=0
    )
    logger.info("image saved: {}".format(os.path.join(path, name)))


def visualize_mini_batch(data, path, name, normalize=True):
    import matplotlib.pyplot as plt
    # order of the data is NCHW and needs to be inverted to NHWC for matplotlib
    data = data.swapaxes(1, 2).swapaxes(2, 3)
    data_size = data.shape[0]
    num_cols = 10
    num_rows = math.ceil(data_size / num_cols)
    fig = plt.figure()
    for idx in range(data_size):
        img = data[idx, :, :, :]
        if normalize:
            img = normalize_image_scale(img)
        plt.subplot(num_rows, num_cols, idx + 1)
        plt.imshow(img, aspect='equal')
        plt.axis('off')
        plt.tight_layout()
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, name), bbox_inches='tight', dpi=32)
