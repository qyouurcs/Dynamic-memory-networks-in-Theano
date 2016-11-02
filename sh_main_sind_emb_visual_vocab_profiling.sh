#!/usr/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: {0} [gpuid=0]"
    exit
fi

gpuid=$1

CUDA_LAUNCH_BLOCKING=1 THEANO_FLAGS="floatX=float32,device=gpu$gpuid" python main_sind_emb_visual_glove.py
