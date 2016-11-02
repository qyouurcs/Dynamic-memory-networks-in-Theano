#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

#THEANO_FLAGS="floatX=float32,device=gpu$gpuid,optimizer=fast_compile" python main_sind_emb.py --batch_size 10
THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_rnn_batch_sind.py --batch_size 10 --load_state ./states_rnn_show_and_tell/dmn_batch_sind.n512.bs60.epoch76.test4.40647.state  --mode test_beam
