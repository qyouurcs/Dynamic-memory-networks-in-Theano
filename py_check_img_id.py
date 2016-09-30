#!/usr/bin/python

import os 
import sys

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <data_split_dir>'.format(sys.argv[0])
        sys.exit()

    split_dir = sys.argv[1]

    dict_imgs = {}
    for root, subdirs, fns in os.walk( os.path.join(split_dir, 'imgs_resized')):

        for fn in fns:
            key = fn.split('_')[0]
            dict_imgs[key] = 1

    
    with open(os.path.join(split_dir, 'annotions.txt')) as fid:
        for aline in fid:
            key = aline.strip().split()[0]
            if key  not in dict_imgs:
                print key
    

