#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

#THEANO_FLAGS="allow_gc=True,optimizer=fast_compile,floatX=float32,device=gpu$gpuid" python main_sind_emb_att_s_rnn_glb.py --save_dir states_emb_att_s_rnn_glb --learning_rate 0.1
THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_emb_att_s_rnn_glb_hard_relu2.py --save_dir states_emb_att_s_rnn_glb_hard_relu2 --learning_rate 0.1 --load_state states_emb_att_s_rnn_glb_hard_relu2/dmn_batch_sind.mh3.n512.bs15.epoch1.test4.92543.state
