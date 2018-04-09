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

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os
import logging
from collections import OrderedDict
import cPickle as pickle

import utils.misc as misc

from core.config import config as cfg
from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2

logger = logging.getLogger(__name__)


# This function looks at all the iters checkpointed and returns latest iter file
def get_checkpoint_resume_file():
    checkpoint_dir = get_checkpoint_directory()
    all_files = os.listdir(checkpoint_dir)
    all_iters = []
    for f in all_files:
        if f.endswith('.pkl') and f.startswith('c2_model_iter'):
            iter_num = int(f.replace('.pkl', '').replace('c2_model_iter', ''))
            all_iters.append(iter_num)
    if len(all_iters) > 0:
        all_iters.sort(reverse=True)
        last_iter = int(all_iters[0])
        filepath = os.path.join(
            checkpoint_dir, 'c2_model_iter{}.pkl'.format(last_iter)
        )
        return filepath
    else:
        return None


# find whether checkpoint exist
def find_checkpoint():
    checkpoint_dir = get_checkpoint_directory()
    checkpointed_files = os.listdir(checkpoint_dir)
    checkpoint_exists = False
    for f in checkpointed_files:
        if f.endswith('.pkl') and f.startswith('c2_model_iter'):
            checkpoint_exists = True
            break
    return checkpoint_exists


def load_model_from_params_file_for_test(model, weights_file):
    logger.info('Initializing from pre-trained file for test...')
    initialize_params_from_file(model=model, weights_file=weights_file)
    return


def load_model_from_params_file(model):
    """
    case 1: CHECKPOINT.RESUME = False and TRAIN.PARAMS_FILE is not none:
        load params_file

    case 2: CHECKPOINT.RESUME = True and TRAIN.PARAMS_FILE is not none:
        case 2a: if checkpoint exist: use checkpoint
        case 2b: if checkpoint not exist: use params_file

    case 3: CHECKPOINT.RESUME = True and TRAIN.PARAMS_FILE is none:
        case 3a: if checkpoint exist: use checkpoint
        case 3b: if checkpoint not exist: set start_model_iter = 0
    """

    use_checkpoint = cfg.CHECKPOINT.RESUME and find_checkpoint()

    if cfg.TRAIN.PARAMS_FILE and not use_checkpoint:
        logger.info('Initializing from pre-trained file...')
        start_model_iter, prev_lr = initialize_params_from_file(
            model=model, weights_file=cfg.TRAIN.PARAMS_FILE,
            load_momentum=False,  # not load momentum if it is pretrained
        )
        logger.info(('Loaded: start_model_iter: {}; prev_lr: {:.8f}').format(
            start_model_iter, prev_lr))
        model.current_lr = prev_lr

        # correct start_model_iter if pretraining uses a different batch size
        # (mainly used for 1-node warmup)
        if cfg.TRAIN.RESUME_FROM_BATCH_SIZE > 0:
            start_model_iter = misc.resume_from(start_model_iter)

        # if we only want the weights
        if cfg.TRAIN.RESET_START_ITER:
            start_model_iter = 0

    elif use_checkpoint:
        logger.info('Initializing from checkpoints...')
        start_model_iter, prev_lr = initialize_params_from_file(
            model=model, weights_file=get_checkpoint_resume_file())
        logger.info(('Loaded: start_model_iter: {}; prev_lr: {:.8f}').format(
            start_model_iter, prev_lr))
        model.current_lr = prev_lr
    else:  # no checkpoint, no params_file
        # Do nothing and return 0
        start_model_iter = 0
        logger.info('No checkpoint found; training from scratch...')

    return start_model_iter


def resume_from(start_model_iter):
    assert(cfg.TRAIN.RESUME_FROM_BATCH_SIZE > 0)
    start_model_iter = int(
        start_model_iter * cfg.TRAIN.RESUME_FROM_BATCH_SIZE /
        cfg.TRAIN.BATCH_SIZE)  # e.g.: 25025 * 256 / 8192
    logger.info('start_model_iter corrected: {}'.format(start_model_iter))
    return start_model_iter


def get_checkpoint_directory():
    if cfg.CHECKPOINT.DIR:
        odir = os.path.abspath(os.path.join(cfg.CHECKPOINT.DIR, 'checkpoints'))
    else:
        raise Exception('No cfg.CHECKPOINT.DIR specified.')

    return odir


def create_and_get_checkpoint_directory():
    checkpoint_dir = get_checkpoint_directory()
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir


# initialize from ImageNet pre-trained weights, and ***inflate*** if necessary
def initialize_master_gpu_model_params(
        model, weights_file, load_momentum=True):
    ws_blobs = workspace.Blobs()
    logger.info("Initializing model params from file: {}".format(weights_file))
    with open(weights_file, 'r') as fopen:
        blobs = pickle.load(fopen)
    if 'blobs' in blobs:
        blobs = blobs['blobs']
    unscoped_blob_names = OrderedDict()

    # Return the model iter from which training should start
    model_iter = 0
    if 'model_iter' in blobs:
        model_iter = blobs['model_iter']

    if 'lr' in blobs:
        prev_lr = float(blobs['lr'])
    elif cfg.TRAIN.RESET_START_ITER:
        prev_lr = 1.
    else:
        raise Exception('No lr blob found.')

    # initialize params, params momentum, computed params
    if 'test' not in model.net.Name() and load_momentum:
        for param in model.params:
            if param in model.TrainableParams():
                unscoped_blob_names[misc.unscope_name(
                    str(param) + '_momentum')] = True
    for blob in model.GetAllParams():
        unscoped_blob_names[misc.unscope_name(str(blob))] = True

    root_gpu_id = cfg.ROOT_GPU_ID
    with core.NameScope('gpu_{}'.format(root_gpu_id)):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, root_gpu_id)):
            for unscoped_blob_name in unscoped_blob_names.keys():
                scoped_blob_name = misc.scoped_name(unscoped_blob_name)
                if unscoped_blob_name not in blobs:
                    logger.info('{:s} not found'.format(unscoped_blob_name))
                    continue
                if scoped_blob_name in ws_blobs:
                    ws_blob = workspace.FetchBlob(scoped_blob_name)

                    if 'pred' in unscoped_blob_name:
                        if np.prod(ws_blob.shape) \
                                != np.prod(blobs[unscoped_blob_name].shape):
                            logger.info(('{:s} (classifier) found but ' +
                                            'unmatching (not loaded):' +
                                            '{} ---> {}')
                                        .format(
                                            unscoped_blob_name,
                                            blobs[unscoped_blob_name].shape,
                                            ws_blob.shape))
                            continue
                        else:
                            blobs[unscoped_blob_name] = np.reshape(
                                blobs[unscoped_blob_name], ws_blob.shape)

                    if len(ws_blob.shape) != \
                            len(blobs[unscoped_blob_name].shape):
                        # inflate if so
                        assert ws_blob.shape[:2] == \
                            blobs[unscoped_blob_name].shape[:2], \
                            ('Workspace blob {} with shape {} does not match '
                             'weights file shape {}').format(
                                unscoped_blob_name, ws_blob.shape,
                                blobs[unscoped_blob_name].shape)
                        assert ws_blob.shape[-2:] == \
                            blobs[unscoped_blob_name].shape[-2:], \
                            ('Workspace blob {} with shape {} does not match '
                             'weights file shape {}').format(
                                unscoped_blob_name, ws_blob.shape,
                                blobs[unscoped_blob_name].shape)

                        logger.info(
                            ('{:s} loaded from weights file into: {:s}' +
                                    ' inflated {} ---> {}').format(
                                unscoped_blob_name, scoped_blob_name,
                                blobs[unscoped_blob_name].shape,
                                ws_blob.shape))
                        # inflate
                        num_inflate = ws_blob.shape[2]
                        blobs[unscoped_blob_name] = np.stack(
                            [blobs[unscoped_blob_name]] * num_inflate,
                            axis=2) / float(num_inflate)
                    else:
                        logger.info(
                            ('{:s} loaded from weights file into: {:s}' +
                                    ' {}').format(
                                unscoped_blob_name, scoped_blob_name,
                                ws_blob.shape))

                    assert ws_blob.shape == blobs[unscoped_blob_name].shape, \
                        ('Workspace blob {} with shape {} does not match '
                         'weights file shape {}').format(
                            unscoped_blob_name, ws_blob.shape,
                            blobs[unscoped_blob_name].shape)
                data = blobs[unscoped_blob_name].astype(np.float32, copy=False)
                workspace.FeedBlob(scoped_blob_name, data)

    # hack fix: load and broadcast lr to all gpus
    for i in range(cfg.NUM_GPUS):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):
            workspace.FeedBlob(
                'gpu_{}/lr'.format(i), np.array(prev_lr, dtype=np.float32))

    return model_iter, prev_lr


def broadcast_parameters(model):
    num_gpus = cfg.NUM_GPUS
    if num_gpus == 1:
        return
    root_gpu_id = cfg.ROOT_GPU_ID
    all_model_params = model.GetAllParams('gpu_{}'.format(root_gpu_id))
    all_params_momentum = []
    if 'test' not in model.net.Name():
        for param in model.GetParams('gpu_{}'.format(root_gpu_id)):
            if param in model.TrainableParams():
                all_params_momentum.append(str(param) + '_momentum')
    all_params = all_model_params + all_params_momentum
    for param in all_params:
        data = workspace.FetchBlob(str(param))
        unscoped_param_name = misc.unscope_name(str(param))
        logger.info('Broadcasting {} to'.format(str(param)))
        for idx in range(root_gpu_id + 1, root_gpu_id + num_gpus):
            with core.NameScope('gpu_{}'.format(idx)):
                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, idx)):
                    gpu_scoped_name = misc.scoped_name(unscoped_param_name)
                    logger.info(' |-> {}'.format(gpu_scoped_name))
                    workspace.FeedBlob(gpu_scoped_name, data)


# initialize the model from a file and broadcast the parameters to all_gpus
# if num_gpus > 1
def initialize_params_from_file(model, weights_file, load_momentum=True):
    logger.info('Initializing model params from file: {}'.format(weights_file))
    model_iter, prev_lr = initialize_master_gpu_model_params(
        model, weights_file, load_momentum
    )
    broadcast_parameters(model)
    return model_iter, prev_lr


def save_model_params(model, params_file, model_iter):
    logger.info("Saving model params to weights file {}".format(params_file))
    root_gpu_id = cfg.ROOT_GPU_ID
    save_params = [str(param) for param in model.GetParams('gpu_{}'.format(
        root_gpu_id))]
    save_computed_params = [
        str(param) for param in model.GetComputedParams('gpu_{}'.format(
            root_gpu_id))]
    save_blobs = {}
    # also save total model iterations so far
    save_blobs['model_iter'] = model_iter + 1
    save_blobs['lr'] = workspace.FetchBlob('gpu_{}/lr'.format(root_gpu_id))
    # save param momentum as well
    for param in save_params:
        if param in model.TrainableParams():
            scoped_blob_name = str(param) + '_momentum'
            unscoped_blob_name = misc.unscope_name(scoped_blob_name)
            if unscoped_blob_name not in save_blobs:
                data = workspace.FetchBlob(scoped_blob_name)
                save_blobs[unscoped_blob_name] = data
                # logger.info(
                #     '{:s} {} -> {:s}'.format(
                #         scoped_blob_name, data.shape,
                #         unscoped_blob_name)
                # )

    for param in save_params + save_computed_params:
        scoped_blob_name = str(param)
        unscoped_blob_name = misc.unscope_name(scoped_blob_name)
        if unscoped_blob_name not in save_blobs:
            data = workspace.FetchBlob(
                scoped_blob_name
            )
            save_blobs[unscoped_blob_name] = data
            # logger.info(
            #     '{:s} {} -> {:s}'.format(
            #         scoped_blob_name, data.shape,
            #         unscoped_blob_name)
            # )
    with open(params_file, 'w') as fwrite:
        pickle.dump(dict(blobs=save_blobs), fwrite, pickle.HIGHEST_PROTOCOL)
