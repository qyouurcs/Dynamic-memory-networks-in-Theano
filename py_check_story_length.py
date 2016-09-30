#!/usr/bin/python
import os
import sys
from collections import defaultdict


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <split_dir>'.format(sys.argv[0])
        sys.exit()
    
    split_dir = sys.argv[1]
    
    dict_story_cnt = {}
    with open(os.path.join(split_dir, 'annotions.txt'),'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            album_id = parts[2]
            if album_id not in dict_story_cnt:
                dict_story_cnt[album_id] = 0

            dict_story_cnt[album_id] += 1
    
    dict_cnt = defaultdict(int)
    for sid in dict_story_cnt:
        dict_cnt[dict_story_cnt[sid]] += 1

    for cnt in dict_cnt:
        print cnt, dict_cnt[cnt]


