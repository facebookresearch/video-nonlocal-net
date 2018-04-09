# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
'''
This script builds a resnet model for video.
This is the modified model architectures used in the git repo:
it is for 8-frame input for "short" video and 32-frame for "long"
'''

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import models.resnet_helper as resnet_helper
from core.config import config as cfg

import logging

logger = logging.getLogger(__name__)

# For more depths, add the block config here
BLOCK_CONFIG = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
}


# ---------------------------------------------------------------------------
# obtain_arc defines the temporal kernel radius and temporal strides for
# each layers residual blocks in a resnet.
# e.g., use_temp_convs = 1 means a temporal kernel of 3 is used.
# In ResNet50, it has (3, 4, 6, 3) blocks in conv2, 3, 4, 5.
# so the lengths of the corresponding lists are (3, 4, 6, 3)
# ---------------------------------------------------------------------------
def obtain_arc(arc_type):

    pool_stride = 1

    # c2d, ResNet50
    if arc_type == 1:
        use_temp_convs_1 = [0]
        temp_strides_1   = [1]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2   = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3   = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 6
        temp_strides_4   = [1, ] * 6
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5   = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    # i3d, ResNet50
    if arc_type == 2:
        use_temp_convs_1 = [2]
        temp_strides_1   = [1]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2   = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3   = [1, 1, 1, 1]
        use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
        temp_strides_4   = [1, 1, 1, 1, 1, 1]
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5   = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    # c2d, ResNet101
    if arc_type == 3:
        use_temp_convs_1 = [0]
        temp_strides_1   = [1]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2   = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3   = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 23
        temp_strides_4   = [1, ] * 23
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5   = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)

    # i3d, ResNet101
    if arc_type == 4:
        use_temp_convs_1 = [2]
        temp_strides_1   = [1]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2   = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3   = [1, 1, 1, 1]
        use_temp_convs_4 = []
        for i in range(23):
            if i % 2 == 0:
                use_temp_convs_4.append(1)
            else:
                use_temp_convs_4.append(0)

        temp_strides_4   = [1, ] * 23
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5   = [1, 1, 1]

        pool_stride = int(cfg.TRAIN.VIDEO_LENGTH / 2)


    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2, use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set   = [temp_strides_1, temp_strides_2, temp_strides_3, temp_strides_4, temp_strides_5]

    return use_temp_convs_set, temp_strides_set, pool_stride


def create_model(model, data, labels, split):
    group = cfg.RESNETS.NUM_GROUPS
    width_per_group = cfg.RESNETS.WIDTH_PER_GROUP
    batch_size = int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)

    logger.info(
        '--------------- ResNet-{} {}x{}d-{}, {} ---------------'.format(
            cfg.MODEL.DEPTH,
            group, width_per_group,
            cfg.RESNETS.TRANS_FUNC,
            cfg.DATASET))

    assert cfg.MODEL.DEPTH in BLOCK_CONFIG.keys(), \
        'Block config is not defined for specified model depth.'
    (n1, n2, n3, n4) = BLOCK_CONFIG[cfg.MODEL.DEPTH]

    res_block = resnet_helper._generic_residual_block_3d
    dim_inner = group * width_per_group

    use_temp_convs_set, temp_strides_set, pool_stride = obtain_arc(cfg.MODEL.VIDEO_ARC_CHOICE)
    print(use_temp_convs_set)
    print(temp_strides_set)


    conv_blob = model.ConvNd(
        data, 'conv1', 3, 64, [1 + use_temp_convs_set[0][0] * 2, 7, 7], strides=[temp_strides_set[0][0], 2, 2],
        pads=[use_temp_convs_set[0][0], 3, 3] * 2,
        weight_init=('MSRAFill', {}),
        bias_init=('ConstantFill', {'value': 0.}), no_bias=1
    )

    test_mode = False if split not in ['test', 'val'] else True
    if cfg.MODEL.USE_AFFINE is False:
        bn_blob = model.SpatialBN(
            conv_blob, 'res_conv1_bn', 64, epsilon=cfg.MODEL.BN_EPSILON,
            momentum=cfg.MODEL.BN_MOMENTUM, is_test=test_mode,
        )
    else:
        bn_blob = model.AffineNd(conv_blob, 'res_conv1_bn', 64)
    relu_blob = model.Relu(bn_blob, bn_blob)
    max_pool = model.MaxPool(relu_blob, 'pool1', kernels=[1, 3, 3], strides=[1, 2, 2], pads=[0, 0, 0] * 2)


    if cfg.MODEL.DEPTH in [50, 101]:

        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            model, res_block, max_pool, 64, 256, stride=1, num_blocks=n1,
            prefix='res2', dim_inner=dim_inner, group=group,
            use_temp_convs=use_temp_convs_set[1], temp_strides=temp_strides_set[1])

        layer_mod = cfg.NONLOCAL.LAYER_MOD
        if cfg.MODEL.DEPTH == 101:
            layer_mod = 2
        if cfg.NONLOCAL.CONV3_NONLOCAL is False:
            layer_mod = 1000

        blob_in = model.MaxPool(blob_in, 'pool2', kernels=[2, 1, 1], strides=[2, 1, 1], pads=[0, 0, 0] * 2)

        if cfg.MODEL.USE_AFFINE is False:
            blob_in, dim_in = resnet_helper.res_stage_nonlocal(
                model, res_block, blob_in, dim_in, 512, stride=2, num_blocks=n2,
                prefix='res3', dim_inner=dim_inner * 2, group=group,
                use_temp_convs=use_temp_convs_set[2], temp_strides=temp_strides_set[2],
                batch_size=batch_size, nonlocal_name='nonlocal_conv3', nonlocal_mod=layer_mod)
        else:
            crop_size = cfg.TRAIN.CROP_SIZE
            blob_in, dim_in = resnet_helper.res_stage_nonlocal_group(
                model, res_block, blob_in, dim_in, 512, stride=2, num_blocks=n2,
                prefix='res3', dim_inner=dim_inner * 2, group=group,
                use_temp_convs=use_temp_convs_set[2], temp_strides=temp_strides_set[2],
                batch_size=batch_size,
                pool_stride=pool_stride, spatial_dim=(int(crop_size / 8)), group_size=4,
                nonlocal_name='nonlocal_conv3', nonlocal_mod=layer_mod)

        layer_mod = cfg.NONLOCAL.LAYER_MOD
        if cfg.MODEL.DEPTH == 101:
            layer_mod = layer_mod * 4 - 1
        if cfg.NONLOCAL.CONV4_NONLOCAL is False:
            layer_mod = 1000

        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            model, res_block, blob_in, dim_in, 1024, stride=2, num_blocks=n3,
            prefix='res4', dim_inner=dim_inner * 4, group=group,
            use_temp_convs=use_temp_convs_set[3], temp_strides=temp_strides_set[3],
            batch_size=batch_size, nonlocal_name='nonlocal_conv4', nonlocal_mod=layer_mod)

        blob_in, dim_in = resnet_helper.res_stage_nonlocal(
            model, res_block, blob_in, dim_in, 2048, stride=2, num_blocks=n4,
            prefix='res5', dim_inner=dim_inner * 8, group=group,
            use_temp_convs=use_temp_convs_set[4], temp_strides=temp_strides_set[4])

    else:
        raise Exception("Unsupported network settings.")

    blob_out = model.AveragePool(blob_in, 'pool5', kernels=[pool_stride, 7, 7], strides=[1, 1, 1], pads=[0, 0, 0] * 2)

    if cfg.TRAIN.DROPOUT_RATE > 0 and test_mode is False:
        blob_out = model.Dropout(
            blob_out, blob_out + '_dropout', ratio=cfg.TRAIN.DROPOUT_RATE, is_test=False)

    if split in ['train', 'val']:
        blob_out = model.FC(
            blob_out, 'pred', dim_in, cfg.MODEL.NUM_CLASSES,
            weight_init=('GaussianFill', {'std': cfg.MODEL.FC_INIT_STD}),
            bias_init=('ConstantFill', {'value': 0.})
        )
    elif split == 'test':
        blob_out = model.ConvNd(
            blob_out, 'pred', dim_in, cfg.MODEL.NUM_CLASSES,
            [1, 1, 1], strides=[1, 1, 1], pads=[0, 0, 0] * 2,
        )

    if split == 'train':
        scale = 1. / cfg.NUM_GPUS
        softmax, loss = model.SoftmaxWithLoss(
            [blob_out, labels], ['softmax', 'loss'], scale=scale)
    elif split == 'val': #in ['test', 'val']:
        softmax = model.Softmax(blob_out, 'softmax', engine='CUDNN')
        loss = None
    elif split == 'test':
        # fully convolutional testing
        blob_out = model.Transpose(blob_out, 'pred_tr', axes=(0, 2, 3, 4, 1,))
        blob_out, old_shape = model.Reshape(
            blob_out, ['pred_re', 'pred_shape5d'],
            shape=(-1, cfg.MODEL.NUM_CLASSES))
        blob_out = model.Softmax(blob_out, 'softmax_conv', engine='CUDNN')
        blob_out = model.Reshape(
            [blob_out, 'pred_shape5d'], ['softmax_conv_re', 'pred_shape2d'])[0]
        blob_out = model.Transpose(blob_out, 'softmax_conv_tr', axes=(0, 4, 1, 2, 3))
        blob_out = model.net.ReduceBackMean(
            [blob_out], ['softmax_ave_w'])
        blob_out = model.ReduceBackMean(
            [blob_out], ['softmax_ave_h'])
        softmax = model.ReduceBackMean(
            [blob_out], ['softmax'])
        loss = None


    return model, softmax, loss
