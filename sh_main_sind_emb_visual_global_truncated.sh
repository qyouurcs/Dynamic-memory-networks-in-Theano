#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_emb_visual_global.py --save_dir states_emb_visual_global --learning_rate 0.1 --load_state states_emb_visual_global/dmn_batch_sind.mh5.n512.bs2.epoch0.test4.71092.state --batch_size 20
