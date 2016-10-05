#!/usr/bin/python
import os
import sys
import lmdb
import numpy as np
import cv2
import caffe
import h5py
import math
import pdb
from caffe.proto import caffe_pb2


if __name__ == '__main__':
    
    if len(sys.argv) < 3:
        print 'Usage: {0} <annotations.txt> <h5_dir>'.format(sys.argv[0])
        sys.exit()
    
    anno_fn = sys.argv[1]
    h5_dir = sys.argv[2]
    max_key_len = 12

    dict_fn2story = {}
    with open(anno_fn, 'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            dict_fn2story[parts[0]] = parts[2]

    if h5_dir[-1] == '/':
        h5_dir = h5_dir[0:-1]

    if not os.path.isdir(h5_dir):
        os.makedirs(h5_dir)

    map_size = 1e12
    item_cnt = 0
    lmdb_name = h5_dir + '_lmdb'

    h5_list = []
    for root, subdirs, fns in os.walk(h5_dir):
        for fn in fns:
            h5_list.append(os.path.join(root,fn))
    
    for idx in range(int(math.ceil(float(len(h5_list))/200))):
        fid_lmdb = lmdb.open(lmdb_name + '_lmdb', map_size = int(map_size))

        with fid_lmdb.begin(write = True) as lmdb_txn:
            start_idx = idx * 200
            end_idx = min((idx + 1) * 200, len(h5_list))
            print 'start_idx = ', start_idx, 'end_idx = ', end_idx

            for i_idx in range(start_idx, end_idx):
                hid = h5py.File(h5_list[i_idx],'r')
                fea = hid['fea'][:]
                fns = hid['fns'][:]
                for fn, f in zip(fns, fea):
                    key = os.path.basename(fn).split('_')[0] 
                    if key not in dict_fn2story:
                        continue

                    key_lmdb = key
                    while len(key_lmdb) < max_key_len:
                        key_lmdb = '0' + key_lmdb
                    datum = caffe.proto.caffe_pb2.Datum()
                    if len(f.shape) == 3:
                        datum.channels = fea.shape[1]
                        datum.height = fea.shape[2]
                        datum.width = fea.shape[3]
                    elif len(f.shape) == 1:
                        datum.channels = fea.shape[1]
                        datum.height = 1
                        datum.width = 1
                    # f: 512 x 14 x 14
                    f_snake = np.zeros_like(f, dtype = 'float32')
                    for i in range(0, f.shape[1], 2):
                        f_snake[:,i,:] = f[:,i,:]
                        f_snake[:,i+1,:] = f[:,i+1,::-1]

                    datum.data = f_snake.tobytes()
                    lmdb_txn.put(key_lmdb.encode('ascii'), datum.SerializeToString())

