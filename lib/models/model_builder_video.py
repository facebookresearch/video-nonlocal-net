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

"""
This script contains functions to build training/testing model abtractions
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
import logging


from caffe2.proto import caffe2_pb2
from caffe2.python import \
    workspace, scope, core, cnn, data_parallel_model

from models import (
    resnet_video,
    resnet_video_org,
)

import utils.misc as misc
import utils.lr_policy as lr_policy

from core.config import config as cfg

logger = logging.getLogger(__name__)

# To add new models, import them, add them to this map and models/TARGETS
model_creator_map = {
    'resnet_video': resnet_video,
    'resnet_video_org': resnet_video_org,
}


class ModelBuilder(cnn.CNNModelHelper):

    def __init__(self, **kwargs):
        kwargs['order'] = 'NCHW'
        self.train = kwargs.get('train', False)
        self.split = kwargs.get('split', 'train')
        self.use_mem_cache = kwargs.get('use_mem_cache', False)
        self.force_fw_only = kwargs.get('force_fw_only', False)

        if 'train' in kwargs:
            del kwargs['train']
        if 'split' in kwargs:
            del kwargs['split']
        if 'use_mem_cache' in kwargs:
            del kwargs['use_mem_cache']
        if 'force_fw_only' in kwargs:
            del kwargs['force_fw_only']

        super(ModelBuilder, self).__init__(**kwargs)
        # Keeping this here in case we have some other params in future to try
        # This is not used for biases anymore
        self.do_not_update_params = []
        self.data_loader = None
        self.input_db = None
        # learning rates
        self.current_lr = 0
        self.SetCurrentLr(0)

        self.model_name = kwargs.get('name')

    def TrainableParams(self, scope=''):
        return [
            param for param in self.params
            if (
                param in self.param_to_grad and   # param has a gradient
                param not in self.do_not_update_params and  # not on blacklist
                (scope == '' or  # filter for gpu assignment, if gpu_id set
                 str(param).find(scope) == 0)
            )
        ]

    def build_model(self, node_id=0):

        # use lmdb
        dirname = cfg.DATADIR
        self.data_loader = self.CreateDB(
            "reader_" + self.split,
            db=dirname + '/' + self.split + '',
            db_type="lmdb",
            num_shards=1,
            shard_id=node_id,
        )

        self.create_data_parallel_model(
            model=self, db_loader=self.data_loader,
            split=self.split, node_id=node_id,
            train=self.train, force_fw_only=self.force_fw_only
        )

    def create_data_parallel_model(
        self, model, db_loader, split, node_id,
        train=True, force_fw_only=False,
    ):
        forward_pass_builder_fun = create_model(model=self, split=split)

        # we use TRAIN.BATCH_SIZE or TEST.BATCH_SIZE as in original img cls code
        batch_size = misc.get_batch_size(self.split)

        # ------- AddVideoInput
        def AddVideoInput(model, reader, batch_size):

            now_height = cfg.TRAIN.JITTER_SCALES[0]
            now_width = int(now_height * 340.0 / 256.0)

            use_mirror = 1
            use_temporal_jitter = 1
            min_size = cfg.TRAIN.JITTER_SCALES[0]
            max_size = cfg.TRAIN.JITTER_SCALES[1]
            is_test = 0
            decode_threads = cfg.VIDEO_DECODER_THREADS
            print(self.model_name)
            if '_bn_aux' in self.model_name:
                decode_threads = 1
            if self.split == 'val':
                decode_threads = 1
            if self.split == 'test':
                use_mirror = 0
                use_temporal_jitter = 0
                min_size = cfg.TEST.SCALE
                max_size = cfg.TEST.SCALE
                is_test = 1

            sample_times = cfg.TEST.NUM_TEST_CLIPS
            # multi crop testing
            if cfg.TEST.USE_MULTI_CROP == 1:
                sample_times = int(sample_times / 3)
            elif cfg.TEST.USE_MULTI_CROP == 2:
                sample_times = int(sample_times / 6)

            data, label = model.net.CustomizedVideoInput(
                reader,
                ["data", "labels"],
                name="data",
                batch_size=batch_size,
                width=now_width,
                height=now_height,
                crop=cfg.TRAIN.CROP_SIZE,  # e.g., 224
                decode_threads=decode_threads,  # e.g., 4
                length=cfg.TRAIN.VIDEO_LENGTH,  # e.g., 32
                sampling_rate=cfg.TRAIN.SAMPLE_RATE,  # e.g., 2
                mirror=use_mirror,
                mean=cfg.MODEL.MEAN,
                std=cfg.MODEL.STD,
                use_local_file=1,
                # for training, we need to random clip
                temporal_jitter=use_temporal_jitter,
                # Note: we set use_scale_augmentaiton = 1 but the range
                # can be fixed, e.g., [256, 256]
                use_scale_augmentaiton=1,
                min_size=min_size,  # e.g., 256
                max_size=max_size,  # e.g., 260
                is_test=is_test,  # make it explicit
                use_bgr=cfg.MODEL.USE_BGR,
                sample_times=sample_times,
                use_multi_crop=cfg.TEST.USE_MULTI_CROP
            )

            data = model.StopGradient(data, data)
        # ------- end of AddVideoInput

        def add_video_input(model):
            AddVideoInput(
                model,
                db_loader,
                batch_size=batch_size,
            )

        input_builder_fun = add_video_input

        if train and not force_fw_only:
            param_update_builder_fun = add_parameter_update_ops(model=model)
        else:
            param_update_builder_fun = None
        first_gpu = cfg.ROOT_GPU_ID
        gpus = range(first_gpu, first_gpu + cfg.NUM_GPUS)

        rendezvous_ctx = None

        data_parallel_model.Parallelize_GPU(
            model,
            input_builder_fun=input_builder_fun,
            forward_pass_builder_fun=forward_pass_builder_fun,
            param_update_builder_fun=param_update_builder_fun,
            devices=gpus,
            rendezvous=rendezvous_ctx,
            broadcast_computed_params=False,
            optimize_gradient_memory=cfg.MODEL.MEMONGER,
            use_nccl=not cfg.DEBUG,  # org: True
        )

    # ----------------------------
    # customized layers
    # ----------------------------
    # relu with inplace option
    def Relu_(self, blob_in):
        blob_out = self.Relu(
            blob_in,
            blob_in if cfg.MODEL.ALLOW_INPLACE_RELU else blob_in + "_relu")
        return blob_out

    def Conv3dBN(
        self, blob_in, prefix, dim_in, dim_out, kernels, strides, pads,
        group=1, bn_init=None,
        **kwargs
    ):
        conv_blob = self.ConvNd(
            blob_in, prefix, dim_in, dim_out, kernels, strides=strides,
            pads=pads, group=group,
            weight_init=("MSRAFill", {}),
            bias_init=('ConstantFill', {'value': 0.}), no_bias=1)
        blob_out = self.SpatialBN(
            conv_blob, prefix + "_bn", dim_out,
            epsilon=cfg.MODEL.BN_EPSILON,
            momentum=cfg.MODEL.BN_MOMENTUM,
            is_test=self.split in ['test', 'val'])

        # set bn init if specified
        if bn_init is not None and bn_init != 1.0:  # numerical issue not matter
            self.param_init_net.ConstantFill(
                [prefix + "_bn_s"],
                prefix + "_bn_s", value=bn_init)

        return blob_out

    # Conv + Affine wrapper
    def Conv3dAffine(  # args in the same order of Conv3d()
        self, blob_in, prefix, dim_in, dim_out, kernels, strides, pads,
        group=1,
        suffix='_bn',
        inplace_affine=False,
        **kwargs
    ):
        conv_blob = self.ConvNd(
            blob_in, prefix, dim_in, dim_out, kernels, strides=strides,
            pads=pads, group=group,
            weight_init=("MSRAFill", {}),
            bias_init=('ConstantFill', {'value': 0.}), no_bias=1)
        blob_out = self.AffineNd(
            conv_blob, prefix + suffix, dim_out, inplace=inplace_affine)

        return blob_out

    # affine
    def AffineNd(
            self, blob_in, blob_out, dim_in, share_with=None, inplace=False):
        blob_out = blob_out or self.net.NextName()
        is_not_sharing = share_with is None
        param_prefix = blob_out if is_not_sharing else share_with
        weight_init = ('ConstantFill', {'value': 1.})
        bias_init = ('ConstantFill', {'value': 0.})
        scale = self.param_init_net.__getattr__(weight_init[0])(
            [],
            param_prefix + '_s',
            shape=[dim_in, ],
            **weight_init[1]
        )
        bias = self.param_init_net.__getattr__(bias_init[0])(
            [],
            param_prefix + '_b',
            shape=[dim_in, ],
            **bias_init[1]
        )
        if is_not_sharing:
            self.net.Proto().external_input.extend([str(scale), str(bias)])
            self.params.extend([scale, bias])
            self.weights.append(scale)
            self.biases.append(bias)
        if inplace:
            return self.net.AffineNd([blob_in, scale, bias], blob_in)
        else:
            return self.net.AffineNd([blob_in, scale, bias], blob_out)

    # ----------------------------
    # learning rate utils
    # ----------------------------
    def SetCurrentLr(self, cur_iter):
        """Set the model's current learning rate without changing any blobs in
        the workspace.
        """
        self.current_lr = lr_policy.get_lr_at_iter(cur_iter)

    def UpdateWorkspaceLr(self, cur_iter):
        """Updates the model's current learning rate and the workspace (learning
        rate and update history/momentum blobs).
        """
        new_lr = lr_policy.get_lr_at_iter(cur_iter)
        if new_lr != self.current_lr:
            # avoid too noisy logging
            if new_lr / self.current_lr < 0.9 or new_lr / self.current_lr > 1.1:
                logger.info(
                    'Setting learning rate to {:.6f} at iteration {}'.format(
                        new_lr, cur_iter))
            self._SetNewLr(new_lr)

    def _SetNewLr(self, new_lr):
        """Do the actual work of updating the model and workspace blobs.
        """
        assert self.current_lr > 0
        for i in range(cfg.NUM_GPUS):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):
                workspace.FeedBlob(
                    'gpu_{}/lr'.format(i), np.array(new_lr, dtype=np.float32))

        if cfg.SOLVER.SCALE_MOMENTUM and \
                abs(new_lr - self.current_lr) > abs(1e-7 * new_lr):
            self._CorrectMomentum(new_lr / self.current_lr)
        self.current_lr = new_lr

    def _CorrectMomentum(self, correction):
        """The MomentumSGDUpdate op implements the update V as

            V := mu * V + lr * grad,

        where mu is the momentum factor, lr is the learning rate, and grad is the
        stochastic gradient. Since V is not defined independently of the learning
        rate (as it should ideally be), when the learning rate is changed we should
        scale the update history V in order to make it compatible in scale with
        lr * grad.
        """
        # avoid too noisy logging
        if correction < 0.9 or correction > 1.1:
            logger.info('Scaling update history by {:.6f} (new/old lr)'.format(
                correction))

        root_gpu_id = cfg.ROOT_GPU_ID
        num_gpus = cfg.NUM_GPUS
        for i in range(root_gpu_id, root_gpu_id + num_gpus):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):
                with core.NameScope("gpu_{}".format(i)):
                    params = self.GetParams()
                    for param in params:
                        if param in self.TrainableParams():
                            op = core.CreateOperator(
                                'Scale', [param + '_momentum'],
                                [param + '_momentum'],
                                scale=correction)
                            workspace.RunOperatorOnce(op)


# ----------------------------
# model utils
# ----------------------------
def create_model(model, split):
    model_name = cfg.MODEL.MODEL_NAME
    assert model_name in model_creator_map.keys(), \
        'Unknown model_type {}'.format(model_name)

    def model_creator(model, loss_scale):
        model, softmax, loss = model_creator_map[model_name].create_model(
            model=model, data="data", labels="labels", split=split,
        )
        return [loss]

    return model_creator


def add_inputs(model, data_loader):
    blob_names = data_loader.get_blob_names()
    queue_name = data_loader._blobs_queue_name

    def input_fn(model):
        for blob_name in blob_names:
            workspace.CreateBlob(scope.CurrentNameScope() + blob_name)
        model.DequeueBlobs(queue_name, blob_names)
        model.StopGradient('data', 'data')

    return input_fn


def add_parameter_update_ops(model):
    def param_update_ops(model):
        lr = model.param_init_net.ConstantFill(
            [], 'lr', shape=[1], value=model.current_lr)
        weight_decay = model.param_init_net.ConstantFill(
            [], 'weight_decay', shape=[1], value=cfg.SOLVER.WEIGHT_DECAY
        )
        weight_decay_bn = model.param_init_net.ConstantFill(
            [], 'weight_decay_bn', shape=[1], value=cfg.SOLVER.WEIGHT_DECAY_BN
        )
        one = model.param_init_net.ConstantFill(
            [], "ONE", shape=[1], value=1.0
        )
        params = model.GetParams()
        curr_scope = scope.CurrentNameScope()
        # scope is of format 'gpu_{}/'.format(gpu_id), so remove the separator
        trainable_params = model.TrainableParams(curr_scope[:-1])
        assert len(params) > 0, 'No trainable params found in model'
        for param in params:
            # only update trainable params
            if param in trainable_params:
                param_grad = model.param_to_grad[param]
                # the param grad is the summed gradient for the parameter across
                # all gpus/hosts
                param_momentum = model.param_init_net.ConstantFill(
                    [param], param + '_momentum', value=0.0)

                if '_bn' in str(param):
                    model.WeightedSum(
                        [param_grad, one, param, weight_decay_bn],
                        param_grad)
                else:
                    model.WeightedSum(
                        [param_grad, one, param, weight_decay],
                        param_grad)
                model.net.MomentumSGDUpdate(
                    [param_grad, param_momentum, lr, param],
                    [param_grad, param_momentum, param],
                    momentum=cfg.SOLVER.MOMENTUM,
                    nesterov=cfg.SOLVER.NESTEROV,
                )
    return param_update_ops
