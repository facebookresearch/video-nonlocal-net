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

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import numpy as np
import datetime
import logging

from core.config import config as cfg
from caffe2.python import workspace

import utils.misc as misc

logger = logging.getLogger(__name__)


class MetricsCalculator():

    def __init__(self, model, split):
        self.model = model
        self.split = split  # 'train', 'val', 'test'
        self.best_top1 = float('inf')
        self.best_top5 = float('inf')
        self.lr = 0  # only used by train
        if cfg.METRICS.EVAL_FIRST_N and split in ['test', 'val']:
            self.best_top1_N_way = float('inf')
            self.best_top5_N_way = float('inf')
        self.reset()

    def reset(self):
        # this should clear out all the metrics computed so far except the
        # best_topN metrics
        logger.info('Resetting {} metrics...'.format(self.split))
        self.aggr_err = 0.0
        self.aggr_err5 = 0.0
        self.aggr_loss = 0.0
        self.aggr_batch_size = 0
        if cfg.METRICS.EVAL_FIRST_N and self.split in ['test', 'val']:
            self.aggr_err_N_way = 0.0
            self.aggr_err5_N_way = 0.0

    def finalize_metrics(self):
        self.avg_err = self.aggr_err / self.aggr_batch_size
        self.avg_err5 = self.aggr_err5 / self.aggr_batch_size
        self.avg_loss = self.aggr_loss / self.aggr_batch_size
        if cfg.METRICS.EVAL_FIRST_N and self.split in ['test', 'val']:
            self.avg_err_N_way = self.aggr_err_N_way / self.aggr_batch_size
            self.avg_err5_N_way = self.aggr_err5_N_way / self.aggr_batch_size

    def get_computed_metrics(self):
        json_stats = {}
        if self.split == 'train':
            json_stats['train_loss'] = self.avg_loss
            json_stats['train_err'] = self.avg_err
            json_stats['train_err5'] = self.avg_err5
        elif self.split in ['test', 'val']:
            json_stats['test_err'] = self.avg_err
            json_stats['test_err5'] = self.avg_err5
            json_stats['best_err'] = self.best_top1
            json_stats['best_err5'] = self.best_top5
            if cfg.METRICS.EVAL_FIRST_N:
                json_stats['test_err_N_way'] = self.avg_err_N_way
                json_stats['test_err5_N_way'] = self.avg_err5_N_way
        return json_stats

    def log_final_metrics(
            self, model_iter, total_iters=cfg.SOLVER.MAX_ITER):
        # 1 / 50000 = 0.002%, so we only need 3 digits for ImageNet val
        if cfg.METRICS.EVAL_FIRST_N:
            print(('* Finished #iters [{}|{}]:' +
                    ' top1: {:.3f} top5: {:.3f}' +
                    ' top1_N_way: {:.3f} top5_N_way: {:.3f}').format(
                model_iter + 1, total_iters,
                self.avg_err, self.avg_err5,
                self.avg_err_N_way, self.avg_err5_N_way
            ))
        else:
            print(('* Finished #iters [{}|{}]:' +
                    ' top1: {:.3f} top5: {:.3f}').format(
                model_iter + 1, total_iters,
                self.avg_err, self.avg_err5,
            ))

    def compute_and_log_best(self):
        if self.avg_err < self.best_top1:
            self.best_top1 = self.avg_err
            self.best_top5 = self.avg_err5
            print('\n* Best model: top1: {:7.3f} top5: {:7.3f}\n'.format(
                self.best_top1, self.best_top5
            ))

        if cfg.METRICS.EVAL_FIRST_N and self.split in ['test', 'val']:
            if self.avg_err_N_way < self.best_top1_N_way:
                self.best_top1_N_way = self.avg_err_N_way
                self.best_top5_N_way = self.avg_err5_N_way
                print('\n* Best: top1_N_way: {:7.3f} top5_N_way: {:7.3f}\n'
                        .format(self.best_top1_N_way, self.best_top5_N_way))

    def calculate_and_log_all_metrics_train(
            self, curr_iter, timer):

        # as sanity check, we always read lr from workspace
        self.lr = float(
            workspace.FetchBlob('gpu_{}/lr'.format(cfg.ROOT_GPU_ID)))

        # as sanity, we only trust what we loaded from workspace
        cur_batch_size = get_batch_size_from_workspace()

        # we only compute loss for train
        cur_loss = sum_multi_gpu_blob('loss')

        self.aggr_loss += cur_loss * cur_batch_size
        self.aggr_batch_size += cur_batch_size

        accuracy_metrics = compute_multi_gpu_topk_accuracy(
            top_k=1, split=self.split)
        accuracy5_metrics = compute_multi_gpu_topk_accuracy(
            top_k=5, split=self.split)

        cur_err = (1.0 - accuracy_metrics['topk_accuracy']) * 100
        cur_err5 = (1.0 - accuracy5_metrics['topk_accuracy']) * 100

        self.aggr_err += cur_err * cur_batch_size
        self.aggr_err5 += cur_err5 * cur_batch_size

        if (curr_iter + 1) % cfg.LOG_PERIOD == 0:
            rem_iters = cfg.SOLVER.MAX_ITER - curr_iter - 1
            eta_seconds = timer.average_time * rem_iters
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            epoch = (curr_iter + 1) \
                / (cfg.TRAIN.DATASET_SIZE / cfg.TRAIN.BATCH_SIZE)

            log_str = ' '.join((
                '| Train ETA: {} LR: {:.8f}',
                ' Iters [{}/{}]',
                '[{:.2f}ep]',
                ' Time {:0.3f}',
                ' Loss {:7.4f}',
                ' top1 {:7.3f} top5 {:7.3f}',
            ))
            print(log_str.format(
                eta, self.lr,
                curr_iter + 1, cfg.SOLVER.MAX_ITER,
                epoch,
                timer.diff,
                cur_loss,
                cur_err, cur_err5,
            ))

    def calculate_and_log_all_metrics_test(self, curr_iter, timer, total_iters):

        # as sanity, we only trust what we loaded from workspace
        cur_batch_size = get_batch_size_from_workspace()

        self.aggr_batch_size += cur_batch_size

        accuracy_metrics = compute_multi_gpu_topk_accuracy(
            top_k=1, split=self.split)
        accuracy5_metrics = compute_multi_gpu_topk_accuracy(
            top_k=5, split=self.split)

        cur_err = (1.0 - accuracy_metrics['topk_accuracy']) * 100
        cur_err5 = (1.0 - accuracy5_metrics['topk_accuracy']) * 100

        self.aggr_err += cur_err * cur_batch_size
        self.aggr_err5 += cur_err5 * cur_batch_size

        # for now, we only compute top N way in test/val
        if len(accuracy_metrics) > 1:
            cur_err_N_way = (
                1.0 - accuracy_metrics['topk_N_way_accuracy']) * 100
            cur_err5_N_way = (
                1.0 - accuracy5_metrics['topk_N_way_accuracy']) * 100
            self.aggr_err_N_way += cur_err_N_way * cur_batch_size
            self.aggr_err5_N_way += cur_err5_N_way * cur_batch_size

        if (curr_iter + 1) % cfg.LOG_PERIOD == 0 \
                or curr_iter + 1 == total_iters:  # we check the last iter

            if cfg.METRICS.EVAL_FIRST_N:
                test_str = ' '.join((
                    '| Test: [{}/{}]',
                    ' Time {:0.3f}',
                    ' top1 {:7.3f} ({:7.3f})',
                    ' top5 {:7.3f} ({:7.3f})'
                    ' top1_N_way {:7.3f} ({:7.3f})',
                    ' top5_N_way {:7.3f} ({:7.3f})',
                    ' current batch {}',
                    ' aggregated batch {}',
                ))
                print(test_str.format(
                    curr_iter + 1, total_iters,
                    timer.diff,
                    cur_err, self.aggr_err / self.aggr_batch_size,
                    cur_err5, self.aggr_err5 / self.aggr_batch_size,
                    cur_err_N_way, self.aggr_err_N_way / self.aggr_batch_size,
                    cur_err5_N_way, self.aggr_err5_N_way / self.aggr_batch_size,
                    cur_batch_size, self.aggr_batch_size
                ))
            else:
                test_str = ' '.join((
                    '| Test: [{}/{}]',
                    ' Time {:0.3f}',
                    ' top1 {:7.3f} ({:7.3f})',
                    ' top5 {:7.3f} ({:7.3f})',
                    ' current batch {}',
                    ' aggregated batch {}',
                ))
                print(test_str.format(
                    curr_iter + 1, total_iters,
                    timer.diff,
                    cur_err, self.aggr_err / self.aggr_batch_size,
                    cur_err5, self.aggr_err5 / self.aggr_batch_size,
                    cur_batch_size, self.aggr_batch_size
                ))


# ----------------------------------------------
# other utils
# ----------------------------------------------
def compute_topk_correct_hits(top_k, preds, labels):
    '''Compute the number of corret hits'''
    batch_size = preds.shape[0]

    top_k_preds = np.zeros((batch_size, top_k), dtype=np.float32)
    for i in range(batch_size):
        top_k_preds[i, :] = np.argsort(-preds[i, :])[:top_k]

    correctness = np.zeros(batch_size, dtype=np.int32)
    for i in range(batch_size):
        if labels[i] in top_k_preds[i, :].astype(np.int32).tolist():
            correctness[i] = 1
    correct_hits = sum(correctness)

    return correct_hits


def compute_multi_gpu_topk_accuracy(top_k, split):

    aggr_batch_size = 0
    aggr_top_k_correct_hits = 0

    if cfg.METRICS.EVAL_FIRST_N and split in ['test', 'val']:
        aggr_top_k_N_way_correct_hits = 0

    computed_metrics = {}

    for idx in range(cfg.ROOT_GPU_ID, cfg.ROOT_GPU_ID + cfg.NUM_GPUS):

        softmax = workspace.FetchBlob('gpu_{}/pred'.format(idx))
        # remove the last two dimensions if we use conv as the output fc
        softmax = softmax.reshape((softmax.shape[0], -1))
        labels = workspace.FetchBlob('gpu_{}/labels'.format(idx))

        assert labels.shape[0] == softmax.shape[0], "Batch size mismatch."

        aggr_batch_size += labels.shape[0]
        aggr_top_k_correct_hits += compute_topk_correct_hits(
            top_k, softmax, labels)

        if cfg.METRICS.EVAL_FIRST_N and split in ['test', 'val']:
            aggr_top_k_N_way_correct_hits += compute_topk_correct_hits(
                top_k, softmax[:, :cfg.METRICS.FIRST_N], labels)

    # normalize results
    computed_metrics['topk_accuracy'] = \
        float(aggr_top_k_correct_hits) / aggr_batch_size
    if cfg.METRICS.EVAL_FIRST_N and split in ['test', 'val']:
        computed_metrics['topk_N_way_accuracy'] = \
            float(aggr_top_k_N_way_correct_hits) / aggr_batch_size

    return computed_metrics


def sum_multi_gpu_blob(blob_name):
    """Summed values of a blob on each gpu"""
    value = 0
    num_gpus = cfg.NUM_GPUS
    root_gpu_id = cfg.ROOT_GPU_ID
    for idx in range(root_gpu_id, root_gpu_id + num_gpus):
        value += workspace.FetchBlob('gpu_{}/{}'.format(idx, blob_name))
    return value


def get_batch_size_from_workspace():
    """Summed values of batch size on each gpu"""
    value = 0
    num_gpus = cfg.NUM_GPUS
    root_gpu_id = cfg.ROOT_GPU_ID
    for idx in range(root_gpu_id, root_gpu_id + num_gpus):
        value += workspace.FetchBlob('gpu_{}/{}'.format(idx, 'pred')).shape[0]
    return value


# ----------------------------------------------
# for logging
# ----------------------------------------------
def get_json_stats_dict(train_meter, test_meter, curr_iter):
    json_stats = dict(
        eval_period=cfg.TRAIN.EVAL_PERIOD,
        batchSize=cfg.TRAIN.BATCH_SIZE,
        dataset=cfg.DATASET,
        num_classes=cfg.MODEL.NUM_CLASSES,
        momentum=cfg.SOLVER.MOMENTUM,
        weightDecay=cfg.SOLVER.WEIGHT_DECAY,
        nGPU=cfg.NUM_GPUS,
        LR=cfg.SOLVER.BASE_LR,
        bn_momentum=cfg.MODEL.BN_MOMENTUM,
        current_learning_rate=train_meter.lr,
    )
    computed_train_metrics = train_meter.get_computed_metrics()
    json_stats.update(computed_train_metrics)
    if test_meter is not None:
        computed_test_metrics = test_meter.get_computed_metrics()
        json_stats.update(computed_test_metrics)

    # other info
    json_stats['used_gpu_memory'] = misc.get_gpu_stats()
    json_stats['currentIter'] = curr_iter + 1
    json_stats['epoch'] = \
        curr_iter / (cfg.TRAIN.DATASET_SIZE / cfg.TRAIN.BATCH_SIZE)

    return json_stats
