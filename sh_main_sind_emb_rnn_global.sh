#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

#THEANO_FLAGS="floatX=float32,device=gpu$gpuid,optimizer=fast_compile" python main_sind_emb_rnn_global.py
THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_emb_rnn_global.py --l2 1e-5 --save_dir  states_rnn_show_global_1e-5
