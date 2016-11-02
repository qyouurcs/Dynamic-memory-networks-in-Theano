#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

#THEANO_FLAGS="floatX=float32,device=gpu$gpuid,optimizer=fast_compile" python main_sind_emb.py --batch_size 10
THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_rnn_batch_sind.py --save_dir states_rnn_show_and_tell --learning_rate 1 --load_state states_rnn_show_and_tell/dmn_batch_sind.n512.bs80.epoch4.test6.07420.state
