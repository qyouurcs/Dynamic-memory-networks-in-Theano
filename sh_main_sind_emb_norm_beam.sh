#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

#THEANO_FLAGS="floatX=float32,device=gpu$gpuid,optimizer=fast_compile" python main_sind_emb.py --batch_size 10
THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_emb.py --batch_size 10 --batch_norm 1 --save_dir states_emb_norm --load_state ./states_emb_norm/dmn_batch_sind.mh5.n512.bs10.bn.epoch6.test3.11125.state --mode test_beam --batch_size 10
