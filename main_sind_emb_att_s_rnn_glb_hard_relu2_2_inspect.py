import sys
import os
import numpy as np
import sklearn.metrics as metrics
import argparse
import time
import json

import utils
import nn_utils

# For logging.
import climate
import pdb
logging = climate.get_logger(__name__)
climate.enable_default_logging()


print "==> parsing input arguments"
parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default="dmn_batch_sind", help='network type: dmn_basic, dmn_smooth, or dmn_batch')
parser.add_argument('--word_vector_size', type=int, default=50, help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--dim', type=int, default=512, help='number of hidden units in input module GRU')
parser.add_argument('--cnn_dim', type=int, default=512, help='number of hidden units in input module GRU')
parser.add_argument('--cnn_dim_fc', type=int, default=4096, help='number of hidden units in input module GRU')
parser.add_argument('--patches', type=int, default=196, help='number of hidden units in input module GRU')
parser.add_argument('--story_len', type=int, default=5, help='number of images in a story')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--load_state', type=str, default="", help='state file path')
parser.add_argument('--answer_module', type=str, default="recurrent", help='answer module type: feedforward or recurrent')
parser.add_argument('--truncate_gradient', type=int, default=5, help='truncate_gradient')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate')

parser.add_argument('--mode', type=str, default="train", help='mode: train or test. Test mode required load_state')
parser.add_argument('--input_mask_mode', type=str, default="sentence", help='input_mask_mode: word or sentence')
parser.add_argument('--memory_hops', type=int, default=3, help='memory GRU steps')
parser.add_argument('--batch_size', type=int, default=15, help='no commment')
parser.add_argument('--data_dir', type=str, default="data/sind", help='data root directory')
parser.add_argument('--save_dir', type=str, default="", help='data root directory')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--normalize_attention', type=bool, default=False, help='flag for enabling softmax on attention vector')
parser.add_argument('--log_every', type=int, default=10, help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=1, help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (between 0 and 1)')
parser.add_argument('--batch_norm', type=bool, default=False, help='batch normalization')
parser.set_defaults(shuffle=True)
args = parser.parse_args()

print args

assert args.word_vector_size in [50, 100, 200, 300]

network_name = args.prefix + '%s.mh%d.n%d.bs%d%s%s%s' % (
    args.network, 
    args.memory_hops, 
    args.dim, 
    args.batch_size, 
    ".na" if args.normalize_attention else "", 
    ".bn" if args.batch_norm else "", 
    (".d" + str(args.dropout)) if args.dropout>0 else ""
    )


if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)
args_dict = dict(args._get_kwargs())
#args_dict['word2vec'] = word2vec
args_dict['word2vec'] = None
    

# init class
if args.network == 'dmn_batch_sind':
    import dmn_batch_sind_emb_att_s_rnn_glb_hard_relu2_2
    dmn = dmn_batch_sind_emb_att_s_rnn_glb_hard_relu2_2.DMN_batch(**args_dict)

else: 
    raise Exception("No such network known: " + args.network)
    

if args.load_state != "":
    dmn.load_state_ins(args.load_state)
