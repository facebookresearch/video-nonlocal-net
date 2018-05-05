
import cPickle as pickle
import numpy as np

model_in = 'input_model_file'
model_out = 'output_model_file'

target_blobs = []
reject_blobs = []

ori_blobs = {}
with open(model_in, 'r') as fopen:
    ori_blobs = pickle.load(fopen)
blobs = ori_blobs['blobs']

del blobs['model_iter']
del_words = ['_riv', '_rm', '_momentum']

print(blobs.keys())
now_keys = blobs.keys()

for i in range(len(now_keys)):
    now_name = now_keys[i]

    flag = 0

    for j in range(len(del_words)):
        if del_words[j] in now_name:
            del blobs[now_name]
            print(now_name)
            flag = 1
            break

    if flag == 1:
        continue

    if 'pred' in now_name:
        del blobs[now_name]
        print(now_name)

blobs['lr'] = 0.01

filehandler = open(model_out, 'wb')
pickle.dump(blobs, filehandler)
filehandler.close()
