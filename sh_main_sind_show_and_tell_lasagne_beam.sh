#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

#THEANO_FLAGS="floatX=float32,device=gpu$gpuid,optimizer=fast_compile" python main_sind_emb.py --batch_size 10
THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_show_and_tell_lasagne.py --mode test_beam --load_state states_show_tell_lasagne_1e-5/dmn_batch_sind.mh5.n512.bs200.bn.epoch64.test4.00984.state --batch_size 10
