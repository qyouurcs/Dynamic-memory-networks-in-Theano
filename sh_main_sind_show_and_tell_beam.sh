#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

#THEANO_FLAGS="floatX=float32,device=gpu$gpuid,optimizer=fast_compile" python main_sind_emb.py --batch_size 10
THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_show_and_tell.py --batch_size 10 --load_state states_show_and_tell/dmn_batch_sind.mh5.n512.bs100.epoch39.test4.76490.state --mode test_beam 
