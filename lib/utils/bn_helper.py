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


# ---------------------------------------------------------------------------
# Kaiming:
# bn_helper is to maintain "faithful" bn stats (not "running" average stats)
# during the training of periods. It computes the true mean/std on a
# sufficiently large training batch, which is then used for test/val.
# "faithful" bn stats are more reliable than "running" stats when we monitor
# the val curves during training, but it often does not improve final results.
# ---------------------------------------------------------------------------

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
import logging

import utils.misc as misc
from utils.timer import Timer

from core.config import config as cfg
from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2

from models import model_builder_video

logger = logging.getLogger(__name__)


class BatchNormHelper():

    def __init__(self):
        self._model = None
        self._bn_layers = None

        self._meanX_dict = {}  # replace "rm"
        self._meanX2_dict = {}

        self._var_dict = {}  # replace "riv"

        self._last_update_iter = -1  # log the last update iter

    def create_bn_aux_model(self, node_id):
        """
        bn_aux_model:
        1. It is like "train", as it uses training data.
        2. It is like "train", as only the "train" mode of bn returns sm/siv
        (sm/siv: the mean and inverse *std* of the current batch)
        3. It is like "val/test", as it does not backprop and does not update
        4. Note: "rm/riv" is fully irrelevant in bn_aux_model
        """
        self._model = model_builder_video.ModelBuilder(
            name='{}_bn_aux'.format(cfg.MODEL.MODEL_NAME),
            train=True,
            use_cudnn=True,
            cudnn_exhaustive_search=True,
            ws_nbytes_limit=(cfg.CUDNN_WORKSPACE_LIMIT * 1024 * 1024),
            split=cfg.TRAIN.DATA_TYPE,
            use_mem_cache=False,  # we don't cache for this
            force_fw_only=True,
        )
        self._model.build_model(node_id=node_id)

        workspace.CreateNet(self._model.net)
        # self._model.start_data_loader()

        misc.save_net_proto(self._model.net)

        self._find_bn_layers()
        self._clean_and_reset_buffer()
        return

    def compute_and_update_bn_stats(self, curr_iter=None):
        """
        We update bn before: (i) testing and (ii) checkpointing.
        They may have different periods.
        To ensure test results are reproducible (not changed by new bn stats),
        We only compute new stats if curr_iter changes.
        """
        if curr_iter is None or curr_iter != self._last_update_iter:
            # start a new compute
            logger.info('Computing and updating BN stats at iter: {}'.format(
                curr_iter + 1))

            self._last_update_iter = curr_iter
            self._clean_and_reset_buffer()

            timer = Timer()
            for i in range(cfg.TRAIN.ITER_COMPUTE_PRECISE_BN):
                timer.tic()
                workspace.RunNet(self._model.net.Proto().name)
                self._collect_bn_stats()
                timer.toc()

                if (i + 1) % cfg.LOG_PERIOD == 0:
                    logger.info('Computing BN [{}/{}]: {:.3}s'.format(
                        i + 1, cfg.TRAIN.ITER_COMPUTE_PRECISE_BN, timer.diff))

            self._finalize_bn_stats()
            self._update_bn_stats_gpu()
        else:
            # use the last compute
            logger.info('BN of iter {} computed. Update to GPU only.'.format(
                curr_iter + 1))
            self._update_bn_stats_gpu()

    def _find_bn_layers(self):
        self._bn_layers = []
        for blob in self._model.params:
            blob = misc.unscope_name(str(blob))
            if blob.endswith('_bn_s'):
                bn_layer = blob[:-5]
                if bn_layer not in self._bn_layers:
                    self._bn_layers.append(bn_layer)

        return

    def _clean_and_reset_buffer(self):
        self._meanX_dict = {}
        self._meanX2_dict = {}
        for bn_layer in self._bn_layers:
            self._meanX_dict[bn_layer] = 0
            self._meanX2_dict[bn_layer] = 0
        return

    def _collect_bn_stats(self):
        """
        # let
        x = workspace.FetchBlob(layername)
        x = x.transpose((1, 0, 2, 3))
        x = x.reshape((x.shape[0], -1))
        # then:
        # sm == np.mean(x, axis=1)
        # siv == 1. / np.sqrt(np.var(x, axis=1) + cfg.MODEL.BN_EPSILON)
        """

        """
        we maintain meanX and meanX2 (X2 = X**2) which are additive
        """
        bn_eps = cfg.MODEL.BN_EPSILON
        num_gpus = cfg.NUM_GPUS
        root_gpu_id = cfg.ROOT_GPU_ID
        for i in range(root_gpu_id, root_gpu_id + num_gpus):
            for bn_layer in self._bn_layers:
                layername = 'gpu_{}/'.format(i) + bn_layer

                single_batch_meanX = workspace.FetchBlob(
                    layername + '_bn_sm')
                single_batch_inv_std = workspace.FetchBlob(
                    layername + '_bn_siv')

                single_batch_var = (1. / single_batch_inv_std) ** 2 - bn_eps
                # var = mean(x ** 2) - mean(x) ** 2
                # np.mean(x ** 2, axis=1) - np.mean(x, axis=1) ** 2
                single_batch_meanX2 = \
                    single_batch_var + single_batch_meanX ** 2

                self._meanX_dict[bn_layer] += single_batch_meanX
                self._meanX2_dict[bn_layer] += single_batch_meanX2

    def _finalize_bn_stats(self):
        """
        update the CPU cache
        """

        normalize = cfg.TRAIN.ITER_COMPUTE_PRECISE_BN * cfg.NUM_GPUS

        self._var_dict = {}

        for bn_layer in self._bn_layers:
            self._meanX_dict[bn_layer] /= normalize
            self._meanX2_dict[bn_layer] /= normalize

            var = self._meanX2_dict[bn_layer] - self._meanX_dict[bn_layer] ** 2

            assert (var > 0.).all(), "layer: {} var < 0".format(bn_layer)
            self._var_dict[bn_layer] = var

        return

    def _update_bn_stats_gpu(self):
        """
        copy to GPU
        note: the actual blobs used at test time are "rm" and "riv"
        """

        num_gpus = cfg.NUM_GPUS
        root_gpu_id = cfg.ROOT_GPU_ID
        for i in range(root_gpu_id, root_gpu_id + num_gpus):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):
                for bn_layer in self._bn_layers:
                    workspace.FeedBlob(
                        'gpu_{}/'.format(i) + bn_layer + '_bn_rm',
                        np.array(self._meanX_dict[bn_layer], dtype=np.float32),
                    )
                    """
                    Note: riv is acutally running var (not running inv var)!!!!
                    """
                    workspace.FeedBlob(
                        'gpu_{}/'.format(i) + bn_layer + '_bn_riv',
                        np.array(self._var_dict[bn_layer], dtype=np.float32),
                    )
        return
