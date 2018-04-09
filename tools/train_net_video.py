# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function


import argparse
import logging
import os
import sys
import math
import numpy as np
import cv2

from caffe2.python import workspace

from core.config import config as cfg
from core.config import (
    cfg_from_file, cfg_from_list, assert_and_infer_cfg, print_cfg)

import utils.checkpoints as checkpoints
import utils.metrics as metrics
import utils.misc as misc
import utils.bn_helper as bn_helper
from utils.timer import Timer

from models import model_builder_video
from test_net_video import test_net

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def create_wrapper(is_train):
    """
    a simpler wrapper that creates the elements for train/test models
    """
    if is_train:
        suffix = '_train'
        split = cfg.TRAIN.DATA_TYPE
        use_mem_cache = cfg.TRAIN.MEM_CACHE
    else:  # is test
        suffix = '_test'.format(cfg.MODEL.MODEL_NAME)
        split = cfg.TEST.DATA_TYPE
        use_mem_cache = True  # we always cache for test

    model = model_builder_video.ModelBuilder(
        name=cfg.MODEL.MODEL_NAME + suffix,
        train=is_train,
        use_cudnn=True,
        cudnn_exhaustive_search=True,
        ws_nbytes_limit=(cfg.CUDNN_WORKSPACE_LIMIT * 1024 * 1024),
        split=split,
        use_mem_cache=use_mem_cache,
    )
    model.build_model()

    if cfg.PROF_DAG:
        model.net.Proto().type = 'prof_dag'
    else:
        model.net.Proto().type = 'dag'

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    # model.start_data_loader()

    timer = Timer()
    meter = metrics.MetricsCalculator(model=model, split=split)

    misc.save_net_proto(model.net)
    misc.save_net_proto(model.param_init_net)

    return model, timer, meter


def train(opts):
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logging.getLogger(__name__)

    assert opts.test_net, "opts.test_net == False is not implemented."

    # generate seed
    misc.generate_random_seed(opts)

    # create checkpoint dir
    checkpoint_dir = checkpoints.create_and_get_checkpoint_directory()
    logger.info('Checkpoint directory created: {}'.format(checkpoint_dir))

    # -------------------------------------------------------------------------
    # build test_model
    # we build test_model first, as we don't want to overwrite init (if any)
    # -------------------------------------------------------------------------
    test_model, test_timer, test_meter = create_wrapper(is_train=False)
    total_test_iters = int(
        math.ceil(cfg.TEST.DATASET_SIZE / float(cfg.TEST.BATCH_SIZE)))
    logger.info('Test iters: {}'.format(total_test_iters))

    # -------------------------------------------------------------------------
    # now, build train_model
    # -------------------------------------------------------------------------
    train_model, train_timer, train_meter = create_wrapper(is_train=True)

    # -------------------------------------------------------------------------
    # build the bn auxilary model (BN, always BN!)
    # -------------------------------------------------------------------------
    if cfg.TRAIN.COMPUTE_PRECISE_BN:
        bn_aux = bn_helper.BatchNormHelper()
        bn_aux.create_bn_aux_model(node_id=opts.node_id)

    # resumed from checkpoint or pre-trained file
    # see checkpoints.load_model_from_params_file for more details
    start_model_iter = 0
    if cfg.CHECKPOINT.RESUME or cfg.TRAIN.PARAMS_FILE:
        start_model_iter = checkpoints.load_model_from_params_file(train_model)

    # -------------------------------------------------------------------------
    # now, start training
    # -------------------------------------------------------------------------
    logger.info("------------- Training model... -------------")
    train_meter.reset()
    last_checkpoint = checkpoints.get_checkpoint_resume_file()

    for curr_iter in range(start_model_iter, cfg.SOLVER.MAX_ITER):
        # set lr
        train_model.UpdateWorkspaceLr(curr_iter)

        # do SGD on 1 training mini-batch
        train_timer.tic()
        workspace.RunNet(train_model.net.Proto().name)
        train_timer.toc()

        test_debug = False
        if test_debug is True:
            save_path = 'temp_save/'
            data_blob = workspace.FetchBlob('gpu_0/data')
            label_blob = workspace.FetchBlob('gpu_0/labels')
            label_blob1 = workspace.FetchBlob('gpu_1/labels')
            data_blob = data_blob * cfg.MODEL.STD + cfg.MODEL.MEAN
            print(label_blob)
            print(label_blob1)
            for i in range(data_blob.shape[0]):
                for j in range(data_blob.shape[2]):
                    temp_img = data_blob[i, :, j, :, :]
                    temp_img = temp_img.transpose([1, 2, 0])
                    temp_img = temp_img.astype(np.uint8)
                    fname = save_path + 'ori_' + str(curr_iter) \
                        + '_' + str(i) + '_' + str(j) + '.jpg'
                    cv2.imwrite(fname, temp_img)

        # show info after iter 1
        if curr_iter == start_model_iter:
            misc.print_net(train_model)
            os.system('nvidia-smi')
            misc.show_flops_params(train_model)

        # check nan
        misc.check_nan_losses()

        if (curr_iter + 1) % cfg.CHECKPOINT.CHECKPOINT_PERIOD == 0 \
                or curr_iter + 1 == cfg.SOLVER.MAX_ITER:
            # --------------------------------------------------------
            # we update bn before testing or checkpointing
            if cfg.TRAIN.COMPUTE_PRECISE_BN:
                bn_aux.compute_and_update_bn_stats(curr_iter)
            # --------------------------------------------------------
            last_checkpoint = os.path.join(
                checkpoint_dir,
                'c2_model_iter{}.pkl'.format(curr_iter + 1))
            checkpoints.save_model_params(
                model=train_model,
                params_file=last_checkpoint,
                model_iter=curr_iter)

        train_meter.calculate_and_log_all_metrics_train(curr_iter, train_timer)

        # --------------------------------------------------------
        # test model
        # --------------------------------------------------------
        if (curr_iter + 1) % cfg.TRAIN.EVAL_PERIOD == 0:
            # we update bn before testing or checkpointing
            if cfg.TRAIN.COMPUTE_PRECISE_BN:
                bn_aux.compute_and_update_bn_stats(curr_iter)

            # start test
            test_meter.reset()
            logger.info("=> Testing model")
            for test_iter in range(0, total_test_iters):
                test_timer.tic()
                workspace.RunNet(test_model.net.Proto().name)
                test_timer.toc()

                test_meter.calculate_and_log_all_metrics_test(
                    test_iter, test_timer, total_test_iters)

            # finishing test
            test_meter.finalize_metrics()
            test_meter.compute_and_log_best()
            test_meter.log_final_metrics(curr_iter)

            # --------------------------------------------------------
            # we finalize and reset train_meter after each test
            train_meter.finalize_metrics()

            json_stats = metrics.get_json_stats_dict(
                train_meter, test_meter, curr_iter)
            misc.log_json_stats(json_stats)

            train_meter.reset()

    if cfg.TRAIN.TEST_AFTER_TRAIN is True:

        # -------------------------------------------------------------------------
        # training finished; test
        # -------------------------------------------------------------------------
        cfg.TEST.PARAMS_FILE = last_checkpoint

        cfg.TEST.OUTPUT_NAME = 'softmax'
        # 10-clip center-crop
        # cfg.TEST.TEST_FULLY_CONV = False
        # test_net()
        # logger.info("10-clip center-crop testing finished")

        # 10-clip spatial fcn
        cfg.TEST.TEST_FULLY_CONV = True
        test_net()
        logger.info("10-clip spatial fcn testing finished")


def main():
    parser = argparse.ArgumentParser(description='Classification model training')
    parser.add_argument('--test_net', type=bool, default=True,
                        help='Test trained model on test data')
    parser.add_argument('--node_id', type=int, default=0,
                        help='Node id')
    parser.add_argument('--config_file', type=str, default=None, required=True,
                        help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default=None, nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)

    assert_and_infer_cfg()
    print_cfg()

    train(args)


if __name__ == '__main__':
    main()
