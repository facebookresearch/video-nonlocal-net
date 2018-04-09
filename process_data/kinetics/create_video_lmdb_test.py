# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import lmdb
import random
import argparse

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace

# this tool allows to create an lmdb database of videos
# which can be loaded from caffe2 VideoInputOp
# INPUT: a list_file contains a list of videos and their labels
# OUTPUT: an lmdb database of videos


def create_an_lmdb_database(list_file, output_file, use_local_file=True):
    print("Write video to a lmdb...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY
    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)

    # start counters
    total_size = 0
    index = 0
    test_start_frame_num = 10

    # initialize empty lists
    list_idx = []
    list_file_name = []
    # list_start_frame = []
    list_label_strings = []

    # read in a list and shuffle
    items = 0
    with open(list_file, 'r') as data:
        for line in data:
            tokens = line.split()
            list_file_name.append(tokens[0])
            # list_start_frame.append(int(tokens[1]))
            list_label_strings.append(tokens[1])
            list_idx.append(items)
            items = items + 1

    with env.begin(write=True) as txn:
        for i in range(items):
            if not use_local_file:
                # read raw video data and store to db
                with open(list_file_name[list_idx[i]], mode='rb') as file:
                    video_data = file.read()
            else:
                # store the full path to local video file
                video_data = list_file_name[list_idx[i]]

            if i % 1000 == 0:
                print(i)


            for j in range(test_start_frame_num):

                tensor_protos = caffe2_pb2.TensorProtos()
                video_tensor = tensor_protos.protos.add()
                video_tensor.data_type = 4  # string data
                video_tensor.string_data.append(video_data)

                label_tensor = tensor_protos.protos.add()
                label_tensor.data_type = 2
                label_string = list_label_strings[list_idx[i]]
                labels = label_string.split(',')
                label_tensor.int32_data.append(i)
                for label in labels:
                    label_tensor.int32_data.append(int(label))

                start_frame_tensor = tensor_protos.protos.add()
                start_frame_tensor.data_type = 2
                start_frame_tensor.int32_data.append(j)

                txn.put(
                    '{}'.format(index).encode('ascii'),
                    tensor_protos.SerializeToString()
                )
                index = index + 1
                total_size = total_size + len(video_data) + sys.getsizeof(int)

    print(
        "Done writing {} clips into database with a total size of {}".format(
            len(list_idx),
            total_size
        )
    )
    return total_size


def main():
    parser = argparse.ArgumentParser(
        description="Caffe2: create video lmdb dataset"
    )
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="Path to write the lmdb database to",
                        required=True)
    parser.add_argument("--list_file", type=str, default=None,
                        help="List file pointing to videos and labels",
                        required=True)

    args = parser.parse_args()
    create_an_lmdb_database(args.list_file, args.dataset_dir)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
