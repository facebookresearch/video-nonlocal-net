# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
This file contains the elementrary functions to build ResNet models.
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

from core.config import config as cfg
import numpy as np
import models.nonlocal_helper as nonlocal_helper


# 3d bottleneck
# 3d conv in the first conv (3x1x1)
def bottleneck_transformation_3d(
        model, blob_in, dim_in, dim_out, stride, prefix, dim_inner, group=1,
        use_temp_conv=1, temp_stride=1):

    conv_op = model.Conv3dAffine if cfg.MODEL.USE_AFFINE else model.Conv3dBN

    # 1x1 layer
    blob_out = conv_op(
        blob_in, prefix + "_branch2a", dim_in, dim_inner,
        [1 + use_temp_conv * 2, 1, 1],
        strides=[temp_stride, 1, 1], pads=[use_temp_conv, 0, 0] * 2,
        inplace_affine=False,
    )
    blob_out = model.Relu_(blob_out)

    # 3x3 layer
    blob_out = conv_op(
        blob_out, prefix + "_branch2b", dim_inner, dim_inner, [1, 3, 3],
        strides=[1, stride, stride], pads=[0, 1, 1] * 2,
        group=group,
        inplace_affine=False,
    )
    blob_out = model.Relu_(blob_out)

    # 1x1 layer (no relu)
    blob_out = conv_op(
        blob_out, prefix + "_branch2c", dim_inner, dim_out, [1, 1, 1],
        strides=[1, 1, 1], pads=[0, 0, 0] * 2,
        inplace_affine=False,  # must be False
        bn_init=cfg.MODEL.BN_INIT_GAMMA)  # revise BN init of the last block

    return blob_out


# shortcut type B
def _add_shortcut_3d(
        model, blob_in, prefix, dim_in, dim_out, stride, temp_stride=1):
    if dim_in == dim_out and temp_stride == 1 and stride == 1:
        # identity mapping (do nothing)
        return blob_in
    else:
        # when dim changes
        conv_op = model.Conv3dAffine if cfg.MODEL.USE_AFFINE else model.Conv3dBN
        return conv_op(
            blob_in, prefix, dim_in, dim_out, [1, 1, 1],
            strides=[temp_stride, stride, stride],
            pads=[0, 0, 0] * 2, group=1,
            inplace_affine=False,)


# residual block abstraction: x + F(x)
def _generic_residual_block_3d(
        model, blob_in, dim_in, dim_out, stride, prefix, dim_inner,
        group=1, use_temp_conv=0, temp_stride=1, trans_func=None):
    # transformation branch (e.g., 1x1-3x3-1x1, or 3x3-3x3), namely, "F(x)"
    if trans_func is None:
        trans_func = globals()[cfg.RESNETS.TRANS_FUNC]

    tr_blob = trans_func(
        model, blob_in, dim_in, dim_out, stride, prefix,
        dim_inner,
        group=group, use_temp_conv=use_temp_conv, temp_stride=temp_stride)

    # creat shortcut, namely, "x"
    sc_blob = _add_shortcut_3d(
        model, blob_in, prefix + "_branch1",
        dim_in, dim_out, stride, temp_stride=temp_stride)

    # addition, namely, "x + F(x)"
    sum_blob = model.net.Sum(
        [tr_blob, sc_blob],  # "tr_blob" goes first to enable inplace
        tr_blob if cfg.MODEL.ALLOW_INPLACE_SUM else prefix + "_sum")

    # relu after addition
    blob_out = model.Relu_(sum_blob)

    return blob_out


def res_stage_nonlocal(
    model, block_fn, blob_in, dim_in, dim_out, stride, num_blocks, prefix,
    dim_inner=None, group=None, use_temp_convs=None, temp_strides=None,
    batch_size=None, nonlocal_name=None, nonlocal_mod=1000,
):
    # prefix is something like: res2, res3, etc.
    # each res layer has num_blocks stacked

    if use_temp_convs is None:
        use_temp_convs = np.zeros(num_blocks).astype(int)
    if temp_strides is None:
        temp_strides = np.ones(num_blocks).astype(int)

    if len(use_temp_convs) < num_blocks:
        for _ in range(num_blocks - len(use_temp_convs)):
            use_temp_convs.append(0)
            temp_strides.append(1)

    for idx in range(num_blocks):
        block_prefix = "{}_{}".format(prefix, idx)
        block_stride = 2 if (idx == 0 and stride == 2) else 1
        blob_in = _generic_residual_block_3d(
            model, blob_in, dim_in, dim_out, block_stride, block_prefix,
            dim_inner, group, use_temp_convs[idx], temp_strides[idx])
        dim_in = dim_out

        if idx % nonlocal_mod == nonlocal_mod - 1:
            blob_in = nonlocal_helper.add_nonlocal(
                model, blob_in, dim_in, dim_in, batch_size,
                nonlocal_name + '_{}'.format(idx), int(dim_in / 2))

    return blob_in, dim_in


def res_stage_nonlocal_group(
    model, block_fn, blob_in, dim_in, dim_out, stride, num_blocks, prefix,
    dim_inner=None, group=None, use_temp_convs=None, temp_strides=None,
    batch_size=None,
    pool_stride=None, spatial_dim=None, group_size=None,
    nonlocal_name=None, nonlocal_mod=1000,
):
    # prefix is something like: res2, res3, etc.
    # each res layer has num_blocks stacked

    if use_temp_convs is None:
        use_temp_convs = np.zeros(num_blocks).astype(int)
    if temp_strides is None:
        temp_strides = np.ones(num_blocks).astype(int)

    if len(use_temp_convs) < num_blocks:
        for _ in range(num_blocks - len(use_temp_convs)):
            use_temp_convs.append(0)
            temp_strides.append(1)

    for idx in range(num_blocks):
        block_prefix = "{}_{}".format(prefix, idx)
        block_stride = 2 if (idx == 0 and stride == 2) else 1
        blob_in = _generic_residual_block_3d(
            model, blob_in, dim_in, dim_out, block_stride, block_prefix,
            dim_inner, group, use_temp_convs[idx], temp_strides[idx])
        dim_in = dim_out

        if idx % nonlocal_mod == nonlocal_mod - 1:
            blob_in = nonlocal_helper.add_nonlocal_group(
                model, blob_in, dim_in, dim_in, batch_size,
                pool_stride, spatial_dim, spatial_dim, group_size,
                nonlocal_name + '_{}'.format(idx), int(dim_in / 2))

    return blob_in, dim_in
