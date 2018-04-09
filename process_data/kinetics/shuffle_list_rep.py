# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import numpy as np
import random

src = 'trainlist.txt'
# outlist = 'trainlist_shuffle_rep.txt'
# outlist = 'trainlist_shuffle_rep2.txt'

assert(len(sys.argv) == 2)
outlist = sys.argv[1]

f = open(src, 'r')
flist = []
for line in f:
    flist.append(line)
f.close()

f2 = open(outlist, 'w')

for t in range(100):

    listlen = len(flist)
    indices = range(listlen)
    random.shuffle(indices)
    print(indices[1:20])

    for i in range(listlen):
        nowid = indices[i]
        line = flist[nowid]
        f2.write(line)


f2.close()
