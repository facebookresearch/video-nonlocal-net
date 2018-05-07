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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from utils.collections import AttrDict

logger = logging.getLogger(__name__)


__C = AttrDict()
config = __C

__C.DEBUG = False

# Training options
__C.DATALOADER = AttrDict()
__C.DATALOADER.MAX_BAD_IMAGES = 100


# Training options
__C.TRAIN = AttrDict()
__C.TRAIN.PARAMS_FILE = b''
__C.TRAIN.DATA_TYPE = b'train'
__C.TRAIN.BATCH_SIZE = 64

# cache images in memory
__C.TRAIN.MEM_CACHE = False

# if the pre-training batchsize does not match the current
__C.TRAIN.RESUME_FROM_BATCH_SIZE = -1
__C.TRAIN.RESET_START_ITER = False

# scale/ar augmeantion
__C.TRAIN.JITTER_SCALES = [256, 480]
__C.TRAIN.CROP_SIZE = 224

# compute precise bn
__C.TRAIN.COMPUTE_PRECISE_BN = True
__C.TRAIN.ITER_COMPUTE_PRECISE_BN = 200

# Number of iterations after which model should be tested on test/val data
__C.TRAIN.EVAL_PERIOD = 5005
__C.TRAIN.DATASET_SIZE = 234643

__C.TRAIN.VIDEO_LENGTH = 32
__C.TRAIN.SAMPLE_RATE = 2
__C.TRAIN.DROPOUT_RATE = 0.0

__C.TRAIN.TEST_AFTER_TRAIN = False


# Train model options
__C.MODEL = AttrDict()
__C.MODEL.NUM_CLASSES = -1
__C.MODEL.MODEL_NAME = b''

# 1: c2d, ResNet50, 2: i3d, ResNet50, 3: c2d, ResNet101, 4: i3d, ResNet101
__C.MODEL.VIDEO_ARC_CHOICE = 1

# arch
__C.MODEL.DEPTH = 50

# bn
__C.MODEL.BN_MOMENTUM = 0.9
__C.MODEL.BN_EPSILON = 1.0000001e-5
# Kaiming:
# We may use 0 to initialize the residual branch of a residual block,
# so the inital state of the block is exactly identiy. This helps optimizaiton.
__C.MODEL.BN_INIT_GAMMA = 1.0

__C.MODEL.FC_INIT_STD = 0.01

__C.MODEL.MEAN = 114.75
__C.MODEL.STD = 57.375

# options to optimize memory usage
__C.MODEL.ALLOW_INPLACE_SUM = True
__C.MODEL.ALLOW_INPLACE_RELU = True  # disable inplace relu for collecting stats
__C.MODEL.ALLOW_INPLACE_RESHAPE = True
__C.MODEL.MEMONGER = True

__C.MODEL.USE_BGR = False  # default is False for historical reason

# when fine-tuning with bn frozen, we turn a bn layer into affine
__C.MODEL.USE_AFFINE = False

__C.MODEL.SAMPLE_THREADS = 8


# for ResNet or ResNeXt only
__C.RESNETS = AttrDict()
__C.RESNETS.NUM_GROUPS = 1
__C.RESNETS.WIDTH_PER_GROUP = 64
__C.RESNETS.STRIDE_1X1 = False
__C.RESNETS.TRANS_FUNC = b'bottleneck_transformation'


# Test
__C.TEST = AttrDict()
__C.TEST.PARAMS_FILE = b''
__C.TEST.DATA_TYPE = b''
__C.TEST.BATCH_SIZE = 64
__C.TEST.TEN_CROP = False  # used by image classification; deprecated
__C.TEST.SCALE = 256
__C.TEST.CROP_SIZE = 224

__C.TEST.OUTPUT_NAME = b'pred'
__C.TEST.TEST_FULLY_CONV = False
__C.TEST.TEST_FULLY_CONV_FLIP = False
__C.TEST.GLOBAL_AVE = False

__C.TEST.DATASET_SIZE = 19761  # size of kinetics test set

# video specific
__C.TEST.VIDEO_LENGTH = 32
__C.TEST.SAMPLE_RATE = 2
__C.TEST.NUM_TEST_CLIPS = 10
# __C.TEST.MULTICLIP_TEST = False
# __C.TEST.NUM_SECTIONS = 2  # original: 1
# __C.TEST.STARTING_CLIP = 0

__C.TEST.USE_MULTI_CROP = 0


# Solver
__C.SOLVER = AttrDict()
__C.SOLVER.NESTEROV = True
__C.SOLVER.WEIGHT_DECAY = 0.0001
__C.SOLVER.WEIGHT_DECAY_BN = 0.0001
__C.SOLVER.MOMENTUM = 0.9

# Learning rates
__C.SOLVER.LR_POLICY = b'steps_with_relative_lrs'
__C.SOLVER.BASE_LR = 0.1
# For imagenet1k, 150150 = 30 epochs for batch size of 256 images over 8 gpus
__C.SOLVER.STEP_SIZES = [150150, 150150, 150150]
__C.SOLVER.LRS = [1, 0.1, 0.01]
# For above batch size, running 100 epochs = 500500 iterations
__C.SOLVER.MAX_ITER = 500500
# to be consistent with detection code, we will turn STEP_SIZES into STEPS
# example: STEP_SIZES [30, 30, 20] => STEPS [0, 30, 60, 80]
__C.SOLVER.STEPS = None
__C.SOLVER.GAMMA = 0.1  # for cfg.SOLVER.LR_POLICY = 'steps_with_decay'

__C.SOLVER.SCALE_MOMENTUM = False

# warmup hack
__C.SOLVER.WARMUP = AttrDict()
__C.SOLVER.WARMUP.WARMUP_ON = False
__C.SOLVER.WARMUP.WARMUP_START_LR = 0.1
__C.SOLVER.WARMUP.WARMUP_END_ITER = 5005 * 5  # 5 epochs

# Checkpoint options
__C.CHECKPOINT = AttrDict()
__C.CHECKPOINT.CHECKPOINT_MODEL = True
__C.CHECKPOINT.CHECKPOINT_PERIOD = -1
__C.CHECKPOINT.RESUME = True
__C.CHECKPOINT.DIR = b'.'


# Non-local Block
__C.NONLOCAL = AttrDict()
__C.NONLOCAL.CONV_INIT_STD = 0.01
__C.NONLOCAL.NO_BIAS = 0
__C.NONLOCAL.USE_MAXPOOL = True
__C.NONLOCAL.USE_SOFTMAX = True
__C.NONLOCAL.USE_ZERO_INIT_CONV = False
__C.NONLOCAL.USE_BN = True
__C.NONLOCAL.USE_SCALE = True
__C.NONLOCAL.USE_AFFINE = False

__C.NONLOCAL.BN_MOMENTUM = 0.9
__C.NONLOCAL.BN_EPSILON = 1.0000001e-5
__C.NONLOCAL.BN_INIT_GAMMA = 0.0

__C.NONLOCAL.LAYER_MOD = 2
__C.NONLOCAL.CONV3_NONLOCAL = True
__C.NONLOCAL.CONV4_NONLOCAL = True


# Metrics option
__C.METRICS = AttrDict()
# For IN5k, we train with the IN5k training set, but we still eval on IN1k val.
# We assume in the 5k-category list, the IN1k categories are the first 1k ones.
# So we use the following options to evaluate the first N-way on val.
__C.METRICS.EVAL_FIRST_N = False  # deprecated
__C.METRICS.FIRST_N = 1000

__C.DATADIR_TRAIN_TIME = \
    b'kinetics/alllist'
__C.DATADIR_TEST_TIME = \
    b'data/kinetics/kinetics_lmdb_gfsai'
__C.FILENAME_GT = \
    b'data/kinetics/val_all_list.txt'

__C.DATADIR = b''
__C.DATASET = b''
__C.ROOT_GPU_ID = 0
__C.CUDNN_WORKSPACE_LIMIT = 256
__C.RNG_SEED = 2
__C.NUM_GPUS = 8

__C.VIDEO_DECODER_THREADS = 4


""" This dir is to cache shared indexing of the datasets.
"""
__C.OUTPUT_DIR = b'gen'

__C.LOG_PERIOD = 10

__C.PROF_DAG = False


def print_cfg():
    import pprint
    logger.info('Training with config:')
    logger.info(pprint.pformat(__C))


def assert_and_infer_cfg():

    # lr schedule
    if __C.SOLVER.STEPS is None:
        # example input: [150150, 150150, 150150]
        __C.SOLVER.STEPS = []
        __C.SOLVER.STEPS.append(0)
        for idx in range(len(__C.SOLVER.STEP_SIZES)):
            __C.SOLVER.STEPS.append(
                __C.SOLVER.STEP_SIZES[idx] + __C.SOLVER.STEPS[idx])
        # now we have [0, 150150, 300300, 450450]

    # we don't want to do 10-crop
    if __C.TEST.TEN_CROP:
        raise Exception('TEST.TEN_CROP is deprecated.')

    assert __C.TRAIN.BATCH_SIZE % __C.NUM_GPUS == 0, \
        "Train batch size should be multiple of num_gpus."

    assert __C.TEST.BATCH_SIZE % __C.NUM_GPUS == 0, \
        "Test batch size should be multiple of num_gpus."


def merge_dicts(dict_a, dict_b):
    from ast import literal_eval
    for key, value in dict_a.items():
        if key not in dict_b:
            raise KeyError('Invalid key in config file: {}'.format(key))
        if type(value) is dict:
            dict_a[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        # the types must match, too
        old_type = type(dict_b[key])
        if old_type is not type(value) and value is not None:
                raise ValueError(
                    'Type mismatch ({} vs. {}) for config key: {}'.format(
                        type(dict_b[key]), type(value), key)
                )
        # recursively merge dicts
        if isinstance(value, AttrDict):
            try:
                merge_dicts(dict_a[key], dict_b[key])
            except BaseException:
                raise Exception('Error under config key: {}'.format(key))
        else:
            dict_b[key] = value


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as fopen:
        yaml_config = AttrDict(yaml.load(fopen))
    merge_dicts(yaml_config, __C)


def cfg_from_list(args_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(args_list) % 2 == 0, 'Specify values or keys for args'
    for key, value in zip(args_list[0::2], args_list[1::2]):
        key_list = key.split('.')
        cfg = __C
        for subkey in key_list[:-1]:
            assert subkey in cfg, 'Config key {} not found'.format(subkey)
            cfg = cfg[subkey]
        subkey = key_list[-1]
        assert subkey in cfg, 'Config key {} not found'.format(subkey)
        try:
            # handle the case when v is a string literal
            val = literal_eval(value)
        except BaseException:
            val = value
        assert isinstance(val, type(cfg[subkey])) or cfg[subkey] is None, \
            'type {} does not match original type {}'.format(
                type(val), type(cfg[subkey]))
        cfg[subkey] = val
