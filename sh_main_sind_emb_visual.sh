#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_emb_visual.py --load_state states_emb_visual/dmn_batch_sind.mh5.n512.bs20.epoch0.test5.76667.state --learning_rate 0.1
