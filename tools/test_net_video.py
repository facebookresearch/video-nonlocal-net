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

import logging
import numpy as np
import argparse
import sys
import pickle
import datetime
import os
import math
import cv2
from sets import Set
from collections import defaultdict

from caffe2.python import workspace

from core.config import config as cfg
from core.config import (
    cfg_from_file, cfg_from_list, assert_and_infer_cfg, print_cfg)
from models import model_builder_video

import utils.misc as misc
import utils.checkpoints as checkpoints
from utils.timer import Timer

FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def test_net_one_section():
    """
    To save test-time memory, we perform multi-clip test in multiple "sections":
    e.g., 10-clip test can be done in 2 sections of 5-clip test
    """
    timer = Timer()
    results = []
    seen_inds = defaultdict(int)

    logger.warning('Testing started...')  # for monitoring cluster jobs
    test_model = model_builder_video.ModelBuilder(
        name='{}_test'.format(cfg.MODEL.MODEL_NAME), train=False,
        use_cudnn=True, cudnn_exhaustive_search=True,
        split=cfg.TEST.DATA_TYPE)

    test_model.build_model()

    if cfg.PROF_DAG:
        test_model.net.Proto().type = 'prof_dag'
    else:
        test_model.net.Proto().type = 'dag'

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net)

    misc.save_net_proto(test_model.net)
    misc.save_net_proto(test_model.param_init_net)

    total_test_net_iters = int(
        math.ceil(float(cfg.TEST.DATASET_SIZE * cfg.TEST.NUM_TEST_CLIPS) / cfg.TEST.BATCH_SIZE))

    if cfg.TEST.PARAMS_FILE:
        checkpoints.load_model_from_params_file_for_test(
            test_model, cfg.TEST.PARAMS_FILE)
    else:
        raise Exception('No params files specified for testing model.')

    for test_iter in range(total_test_net_iters):
        timer.tic()
        workspace.RunNet(test_model.net.Proto().name)
        timer.toc()

        if test_iter == 0:
            misc.print_net(test_model)
            os.system('nvidia-smi')

        test_debug = False
        if test_debug is True:
            save_path = 'temp_save/'
            data_blob = workspace.FetchBlob('gpu_0/data')
            label_blob = workspace.FetchBlob('gpu_0/labels')
            print(label_blob)
            data_blob = data_blob * cfg.MODEL.STD + cfg.MODEL.MEAN
            for i in range(data_blob.shape[0]):
                for j in range(4):
                    temp_img = data_blob[i, :, j, :, :]
                    temp_img = temp_img.transpose([1, 2, 0])
                    temp_img = temp_img.astype(np.uint8)
                    fname = save_path + 'ori_' + str(test_iter) \
                        + '_' + str(i) + '_' + str(j) + '.jpg'
                    cv2.imwrite(fname, temp_img)

        video_ids_list = []  # for logging
        for gpu_id in range(cfg.NUM_GPUS):
            prefix = 'gpu_{}/'.format(gpu_id)

            softmax_gpu = workspace.FetchBlob(prefix + cfg.TEST.OUTPUT_NAME)
            softmax_gpu = softmax_gpu.reshape((softmax_gpu.shape[0], -1))
            # This is the index of the video for recording results, not the actual class label for the video 
            video_id_gpu = workspace.FetchBlob(prefix + 'labels')

            for i in range(len(video_id_gpu)):
                seen_inds[video_id_gpu[i]] += 1

            video_ids_list.append(video_id_gpu[0])
            # print(video_id_gpu)

            # collect results
            for i in range(softmax_gpu.shape[0]):
                probs = softmax_gpu[i].tolist()
                vid = video_id_gpu[i]
                if seen_inds[vid] > cfg.TEST.NUM_TEST_CLIPS:
                    logger.warning('Video id {} have been seen. Skip.'.format(
                        vid,))
                    continue

                save_pairs = [vid, probs]
                results.append(save_pairs)

        # ---- log
        eta = timer.average_time * (total_test_net_iters - test_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta)))
        logger.info(('{}/{} iter ({}/{} videos):' +
                    ' Time: {:.3f} (ETA: {}). ID: {}').format(
                        test_iter, total_test_net_iters,
                        len(seen_inds), cfg.TEST.DATASET_SIZE,
                        timer.diff, eta,
                        video_ids_list,))

    return results


def test_net():
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    np.random.seed(cfg.RNG_SEED)

    cfg.TEST.DATA_TYPE = 'test'
    if cfg.TEST.TEST_FULLY_CONV is True:
        cfg.TRAIN.CROP_SIZE = cfg.TRAIN.JITTER_SCALES[0]
        cfg.TEST.USE_MULTI_CROP = 1
    elif cfg.TEST.TEST_FULLY_CONV_FLIP is True:
        cfg.TRAIN.CROP_SIZE = cfg.TRAIN.JITTER_SCALES[0]
        cfg.TEST.USE_MULTI_CROP = 2
    else:
        cfg.TRAIN.CROP_SIZE = 224

    # ------------------------------------------------------------------------
    logger.info('Setting test crop_size to: {}'.format(
        cfg.TRAIN.CROP_SIZE))

    print_cfg()
    # ------------------------------------------------------------------------

    results = []
    workspace.ResetWorkspace()  # for memory
    logger.info("Done ResetWorkspace...")

    results = test_net_one_section()

    # evaluate
    if cfg.FILENAME_GT is not None:
        evaluate_result(results)

    # save temporary file
    pkl_path = os.path.join(cfg.CHECKPOINT.DIR, "results_probs.pkl")

    with open(pkl_path, 'w') as f:
        pickle.dump(results, f)
    logger.info('Temporary file saved to: {}'.format(pkl_path))


def read_groundtruth(filename_gt):
    f = open(filename_gt, 'r')
    labels = []
    for line in f:
        rows = line.split()
        labels.append(int(rows[1]))
    f.close()
    return labels


def evaluate_result(results):
    gt_labels = read_groundtruth(cfg.FILENAME_GT)

    sample_num = cfg.TEST.DATASET_SIZE
    class_num = cfg.MODEL.NUM_CLASSES
    sample_video_times = cfg.TEST.NUM_TEST_CLIPS
    counts = np.zeros(sample_num, dtype=np.int32)
    probs = np.zeros((sample_num, class_num))

    assert(len(gt_labels) == sample_num)

    """
    clip_accuracy: the (e.g.) 10*19761 clips' average accuracy
    clip1_accuracy: the 1st clip's accuracy (starting from frame 0)
    """
    clip_accuracy = 0
    clip1_accuracy = 0
    clip1_count = 0
    seen_inds = defaultdict(int)

    # evaluate
    for entry in results:
        vid = entry[0]
        prob = np.array(entry[1])
        probs[vid] += prob[0: class_num]
        counts[vid] += 1

        idx = prob.argmax()
        if idx == gt_labels[vid]:
            # clip accuracy
            clip_accuracy += 1

        # clip1 accuracy
        seen_inds[vid] += 1
        if seen_inds[vid] == 1:
            clip1_count += 1
            if idx == gt_labels[vid]:
                clip1_accuracy += 1

    # sanity checkcnt = 0
    max_clips = 0
    min_clips = sys.maxsize
    count_empty = 0
    count_corrupted = 0
    for i in range(sample_num):
        max_clips = max(max_clips, counts[i])
        min_clips = min(min_clips, counts[i])
        if counts[i] != sample_video_times:
            count_corrupted += 1
            logger.warning('Id: {} count: {}'.format(i, counts[i]))
        if counts[i] == 0:
            count_empty += 1

    logger.info('Num of empty videos: {}'.format(count_empty))
    logger.info('Num of corrupted videos: {}'.format(count_corrupted))
    logger.info('Max num of clips in a video: {}'.format(max_clips))
    logger.info('Min num of clips in a video: {}'.format(min_clips))

    # clip1 accuracy for sanity (# print clip1 first as it is lowest)
    logger.info('Clip1 accuracy: {:.2f} percent ({}/{})'.format(
        100. * clip1_accuracy / clip1_count, clip1_accuracy, clip1_count))

    # clip accuracy for sanity
    logger.info('Clip accuracy: {:.2f} percent ({}/{})'.format(
        100. * clip_accuracy / len(results), clip_accuracy, len(results)))

    # compute accuracy
    accuracy = 0
    accuracy_top5 = 0

    for i in range(sample_num):
        prob = probs[i]

        # top-1
        idx = prob.argmax()
        if idx == gt_labels[i] and counts[i] > 0:
            accuracy = accuracy + 1

        ids = np.argsort(prob)[::-1]
        for j in range(5):
            if ids[j] == gt_labels[i] and counts[i] > 0:
                accuracy_top5 = accuracy_top5 + 1
                break

    accuracy = float(accuracy) / float(sample_num)
    accuracy_top5 = float(accuracy_top5) / float(sample_num)

    logger.info('-' * 80)
    logger.info('top-1 accuracy: {:.2f} percent'.format(accuracy * 100))
    logger.info('top-5 accuracy: {:.2f} percent'.format(accuracy_top5 * 100))
    logger.info('-' * 80)


def main():
    parser = argparse.ArgumentParser(description='Classification model testing')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional config file for params')
    parser.add_argument('opts', help='see configs.py for all options',
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

    test_net()


if __name__ == '__main__':
    main()
