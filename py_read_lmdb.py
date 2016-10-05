import numpy as np
import lmdb
import caffe
import sys
import os
import pdb

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <lmdb_db> <annotations.txt>'.format(sys.argv[0])
        sys.exit()
   
    lmdb_db = sys.argv[1]
    anno_fn = sys.argv[2] 
    max_len = 12
    
    env = lmdb.open(lmdb_db, readonly=True)
    with env.begin() as txn:
        with open(anno_fn,'r') as fid:
            for aline in fid:
                parts = aline.strip().split()
                key = parts[0]
                
                while len(key) < max_len:
                    key = '0' + key
                print key
                raw_datum = txn.get(key.encode('ascii'))
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(raw_datum)
                
                flat_x = np.fromstring(datum.data, dtype=np.float32)
                x = flat_x.reshape(datum.channels, datum.height, datum.width)
                
                print x.shape
