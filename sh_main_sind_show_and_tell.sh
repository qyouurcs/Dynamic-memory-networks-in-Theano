#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

#THEANO_FLAGS="floatX=float32,device=gpu$gpuid,optimizer=fast_compile" python main_sind_emb.py --batch_size 10
THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_show_and_tell.py --save_dir states_show_and_tell
