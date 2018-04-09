# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys
import cPickle as pickle
import numpy as np
import convert_to_finetune

assert(len(sys.argv) == 3)

# model_in  = '../../data/checkpoints/run_c2d_baseline/c2_model_iter300000.pkl'
# model_out = '../../data/pretrained_model/run_c2d_baseline/affine_model_30k.pkl'

# model_in  = '../../data/checkpoints/run_i3d_baseline/c2_model_iter300000.pkl'
# model_out = '../../data/pretrained_model/run_i3d_baseline/affine_model_30k.pkl'

model_in = sys.argv[1]
model_out = sys.argv[2]

ori_blobs = convert_to_finetune.load_and_convert_caffe2_cls_model(model_in)
blobs = ori_blobs['blobs']
now_keys = blobs.keys()
print(now_keys)

for i in range(len(now_keys)):
    now_name = now_keys[i]
    if 'pred' in now_name:
        del blobs[now_name]
        print(now_name)
    if 'momentum' in now_name:
        del blobs[now_name]
        print(now_name)
blobs['lr'] = 0.00125


filehandler = open(model_out, 'wb')
pickle.dump(blobs, filehandler)
filehandler.close()
