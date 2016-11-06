#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0] [mode=train]"
    exit
fi

gpuid=$1
mode='train'
if [ $# -gt 1 ]; then
    mode=$2
fi


if [[ $mode == 'train' ]]; then
    THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_emb_rnn_global.py --l2 1e-5 --save_dir  states_rnn_show_global_1e-5
else

    THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_emb_rnn_global.py --batch_size 10 --save_dir states_rnn_show_global_1e-5 --load_state ./states_rnn_show_global_1e-5/dmn_batch_sind.n512.bs10.epoch70.test4.55731.state --mode test_beam 
fi

#THEANO_FLAGS="floatX=float32,device=gpu$gpuid,optimizer=fast_compile" python main_sind_emb_rnn_global.py
