#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

#THEANO_FLAGS="floatX=float32,device=gpu$gpuid,optimizer=fast_compile" python main_sind_emb.py --load_state ./states_emb/dmn_batch_sind.mh5.n512.bs100.epoch100.test3.79743.state --mode test_beam --batch_size 10
THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_emb.py --load_state states_emb_1e-5/dmn_batch_sind.mh5.n512.bs100.epoch170.test4.95908.state --mode test_beam --batch_size 10
