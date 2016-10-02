#!/usr/bin/python
import os
import sys
from collections import defaultdict

if __name__ == '__main__':

    if len(sys.argv) < 2:
        print 'Usage: {0} <annotations.txt> <glove.txt>'.format(sys.argv[0])
        sys.exit()

    vocab_fn = sys.argv[1]
    glove_fn = sys.argv[2]

    
    dict_glove = {}
    with open(glove_fn,'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            dict_glove[parts[0]] = 0
    
    dict_words = defaultdict(int)
    with open(vocab_fn,'r') as fid:
        for aline in fid:
            parts = aline.strip().split()
            words = parts[5:]
            for word in words:
                if word not in dict_glove:
                    dict_words[word] += 1
    
    for word in dict_words:
        print word, dict_words[word]
