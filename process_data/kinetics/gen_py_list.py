# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import numpy as np
import os
import shutil

srclist = 'classids.json'

videodir = 'YOUR_DATASET_FOLDER/train/'
outlist = 'trainlist.txt'

# videodir = 'YOUR_DATASET_FOLDER/val/'
# outlist = 'vallist.txt'



f = open(outlist, 'w')


json_data = open(srclist).read()
clss_ids = json.loads(json_data)

folder_list = os.listdir(videodir)

was = np.zeros(1000)

for i in range(len(folder_list)):

    folder_name = folder_list[i]
    folder_name_v1 = videodir + folder_name

    names = folder_name.split(' ')
    key_name = folder_name
    if len(names) > 1:
        key_name = '\"' + folder_name + '\"'

    lbl = clss_ids[key_name]
    newfolder_name = ''

    for j in range(len(names)):
        newfolder_name = newfolder_name + names[j]
        if j < len(names) - 1:
            newfolder_name = newfolder_name + '_'

    print(newfolder_name)
    folder_name_cmb = videodir + newfolder_name

    shutil.move(folder_name_v1, folder_name_cmb)

    was[lbl] = 1

    video_list = os.listdir(folder_name_cmb)

    for j in range(len(video_list)):
        video_file = video_list[j]
        video_file = folder_name_cmb + '/' + video_file
        f.write(video_file + ' ' + str(lbl) + '\n')


f.close()

cnt = 0
for i in range(400):
    cnt = cnt + was[i]

print(cnt)
