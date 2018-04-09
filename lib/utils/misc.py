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

import os
import logging
import json
import math
import numpy as np
import subprocess
# import fbcaffe2.tensorboard as tb
from caffe2.python import workspace, scope
from core.config import config as cfg

logger = logging.getLogger(__name__)


def check_nan_losses():
    num_gpus = cfg.NUM_GPUS
    # if any of the losses is NaN, raise exception
    for idx in range(num_gpus):
        loss = workspace.FetchBlob('gpu_{}/loss'.format(idx + cfg.ROOT_GPU_ID))
        if math.isnan(loss):
            logger.error("ERROR: NaN losses on gpu_{}".format(idx))
            os._exit(0)


def get_model_proto_directory():
    odir = os.path.abspath(os.path.join(cfg.CHECKPOINT.DIR))
    if not os.path.exists(odir):
        os.makedirs(odir)
    return odir


def get_num_test_iter(db):
    if cfg.TEST.TEN_CROP:
        raise NotImplementedError('Deprecated: 10-crop test.')
    else:
        # we don't support rounded mini-batch length for deterministic results
        assert(db.get_db_size() % cfg.TEST.BATCH_SIZE == 0)
        test_epoch_iter = int(
            math.ceil(db.get_db_size() / float(cfg.TEST.BATCH_SIZE)))
    return test_epoch_iter


def get_batch_size(split):
    if split in ['test', 'val']:
        if cfg.TEST.TEN_CROP:
            raise NotImplementedError('Deprecated: 10-crop test.')
        else:
            # for test, always specify 1-node batchsize
            batch_size = int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS)
        return batch_size
    elif split == 'train':
        batch_size = int(
            cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS
        )
        return batch_size


def log_json_stats(stats):
    print('\njson_stats: {:s}\n'.format(json.dumps(stats, sort_keys=False)))
    return None


def save_net_proto(net):
    net_proto = str(net.Proto())
    net_name = net.Proto().name
    proto_path = get_model_proto_directory()
    net_proto_path = os.path.join(proto_path, net_name + ".pbtxt")
    with open(net_proto_path, 'w') as wfile:
        wfile.write(net_proto)
    logger.info("{}: Net proto saved to: {}".format(net_name, net_proto_path))
    return None


def unscope_name(blob_name):
    return blob_name[blob_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]


def scoped_name(blob_name):
    return scope.CurrentNameScope() + blob_name


def print_model_param_shape(model):
    all_blobs = model.GetParams('gpu_{}'.format(cfg.ROOT_GPU_ID))
    logger.info('All blobs in workspace:\n{}'.format(all_blobs))
    for blob_name in all_blobs:
        blob = workspace.FetchBlob(blob_name)
        logger.info("{} -> {}".format(blob_name, blob.shape))


def print_buf_net(net):
    logger.info("\n\nPrinting Model: {}\n\n".format(net.Name()))
    master_gpu = 'gpu_{}'.format(cfg.ROOT_GPU_ID)
    op_output = net.Proto().op
    for idx in range(len(op_output)):
        input_b = net.Proto().op[idx].input
        # for simplicity: only print the first output;
        output_b = str(net.Proto().op[idx].output[0])
        type_b = net.Proto().op[idx].type
        if output_b.find(master_gpu) >= 0:
            # Only print the forward pass network
            output_shape = np.array(workspace.FetchBlob(str(output_b))).shape
            first_blob = True
            suffix = ' ------- (op: {:s})'.format(type_b)
            for j in range(len(input_b)):
                input_shape = np.array(
                    workspace.FetchBlob(str(input_b[j]))).shape
                if input_shape != ():
                    logger.info(
                        '{:36s}: {:28s} => {:36s}: {:28s}{}'.format(
                            unscope_name(str(input_b[j])),
                            '{}'.format(input_shape),  # suppress warning
                            unscope_name(str(output_b)),
                            '{}'.format(output_shape),
                            suffix
                        ))
                    if first_blob:
                        first_blob = False
                        suffix = ' ------|'
    logger.info("End of net: {}\n\n".format(net.Name()))


def print_net(model):
    logger.info("Printing Model: {}".format(model.net.Name()))
    master_gpu = 'gpu_{}'.format(cfg.ROOT_GPU_ID)
    op_output = model.net.Proto().op
    model_params = model.GetAllParams(master_gpu)
    for idx in range(len(op_output)):
        input_b = model.net.Proto().op[idx].input
        # for simplicity: only print the first output;
        # not recommended if there are split layers.
        output_b = str(model.net.Proto().op[idx].output[0])
        type_b = model.net.Proto().op[idx].type
        if output_b.find(master_gpu) >= 0:
            # Only print the forward pass network
            if output_b.find('grad') >= 0:
                break
            output_shape = np.array(workspace.FetchBlob(str(output_b))).shape
            first_blob = True
            suffix = ' ------- (op: {:s})'.format(type_b)
            for j in range(len(input_b)):
                if input_b[j] in model_params:
                        continue
                input_shape = np.array(
                    workspace.FetchBlob(str(input_b[j]))).shape
                if input_shape != ():
                    logger.info(
                        '{:28s}: {:20s} => {:36s}: {:20s}{}'.format(
                            unscope_name(str(input_b[j])),
                            '{}'.format(input_shape),  # suppress warning
                            unscope_name(str(output_b)),
                            '{}'.format(output_shape),
                            suffix
                        ))
                    if first_blob:
                        first_blob = False
                        suffix = ' ------|'
    logger.info("End of model: {}".format(model.net.Name()))


def get_gpu_stats():
    sp = subprocess.Popen(
        ['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str = sp.communicate()
    out_list = out_str[0].split('\n')
    out_dict = {}
    for item in out_list:
        try:
            key, val = item.split(':')
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except Exception:
            pass
    used_gpu_memory = out_dict['Used GPU Memory']
    return used_gpu_memory


def show_flops_params(model):
    model_flops, model_params = get_flops_params(model)
    logger.info('Total conv/fc/matMul FLOPs: {}(e9)'.format(model_flops / 1e9))
    logger.info('Total conv/fc params: {}(e6)'.format(model_params / 1e6))


def get_flops_params(model):
    model_ops = model.net.Proto().op
    master_gpu = 'gpu_{}'.format(cfg.ROOT_GPU_ID)
    param_ops = []
    for idx in range(len(model_ops)):
        op_type = model.net.Proto().op[idx].type
        op_input = model.net.Proto().op[idx].input[0]
        if op_type in ['Conv', 'FC', 'BatchMatMul'] and op_input.find(master_gpu) >= 0:
            param_ops.append(model.net.Proto().op[idx])

    num_flops = 0
    num_params = 0
    for idx in range(len(param_ops)):
        op = param_ops[idx]
        op_type = op.type
        op_inputs = param_ops[idx].input
        op_output = param_ops[idx].output[0]
        layer_flops = 0
        layer_params = 0
        if op_type == 'Conv':
            for op_input in op_inputs:
                if '_w' in op_input:
                    param_blob = op_input
                    param_shape = np.array(
                        workspace.FetchBlob(str(param_blob))).shape
                    layer_params = np.prod(param_shape)
                    output_shape = np.array(
                        workspace.FetchBlob(str(op_output))).shape
                    layer_flops = layer_params * np.prod(output_shape[2:])
        elif op_type == 'FC':
            for op_input in op_inputs:
                if '_w' in op_input:
                    param_blob = op_input
                    param_shape = np.array(
                        workspace.FetchBlob(str(param_blob))).shape
                    layer_params = np.prod(param_shape)
                    layer_flops = layer_params
        elif op_type == 'BatchMatMul':
            blob_params = []
            for op_input in op_inputs:
                param_shape = np.array(workspace.FetchBlob(str(op_input))).shape
                blob_params.append(param_shape[1])
                blob_params.append(param_shape[2])
            if 'grad' in op_inputs[0] or 'grad' in op_inputs[1]:
                continue
            if 'shared' in op_inputs[0] or 'shared' in op_inputs[1]:
                continue
            blob_params = np.array(blob_params)
            blob_params = np.unique(blob_params)
            if len(blob_params) == 3:
                layer_flops = blob_params[0] * blob_params[1] * blob_params[2]
            elif len(blob_params) == 1:
                layer_flops = blob_params[0] * blob_params[0] * blob_params[0]
            else:
                layer_flops = 0
                print('confused with matmul dimensions, ignore it for now')

        num_flops += layer_flops
        num_params += layer_params
    return num_flops, num_params


def generate_random_seed(node_id=0):
    np.random.seed(cfg.RNG_SEED)
