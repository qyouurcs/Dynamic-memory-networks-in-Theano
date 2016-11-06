#!/usr/bin/python

import os
import sys
import json
import pdb

# This one will ext from json and then sort according to the filtered as well as output the filtered as reference.

if __name__ == '__main__':

    if len(sys.argv) < 3:
        print 'Usage: {0} <beam.json> <annotations.txt>'.format(sys.argv[0])
        sys.exit()

    json_fn = sys.argv[1]
    anno_fn = sys.argv[2]

    save_dir = json_fn + '_for_metric'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    dict_id2sid = {}
    dict_key2s = {}
    with open(anno_fn, 'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            iid = parts[0]
            sid = parts[2]
            #dict_key2s[sid] = ' '.join(parts[4:])
            if sid not in dict_key2s:
                dict_key2s[sid] = []
            dict_key2s[sid].append((iid, ' '.join(parts[4:])))
            dict_id2sid[iid] = sid

    json_fid = open(json_fn,'r')
    list_obj = json.load(json_fid)
    
    dict_key2s_p = {}
    for obj in list_obj:
        iid = obj['image_id'].split('_')[1]
        sid = obj['image_id'].split('_')[0]

        if sid not in dict_key2s_p:
            dict_key2s_p[sid] = []

        dict_key2s_p[sid].append((iid, obj['caption']))

    
    with open(os.path.join(save_dir, 'pred.txt'),'w') as pfid:
        with open(os.path.join(save_dir, 'ref.txt'),'w') as rfid:
            for sid in dict_key2s_p:
                if len(dict_key2s_p[sid]) == len(dict_key2s[sid]):
                   d = {}
                   for p in dict_key2s_p[sid]:
                       d[p[0]] = p[1]
                   for r in dict_key2s[sid]:
                       rfid.write(r[0] + ' ' + sid + ' ' + r[1] +'\n')
                       pdb.set_trace()
                       pfid.write(r[0] + ' ' + sid + ' ' + d[r[0]] +'\n')

    print 'Done with', os.path.join(save_dir, 'pred.txt')
    print 'Done with', os.path.join(save_dir, 'ref.txt')
