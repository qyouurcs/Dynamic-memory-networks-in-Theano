'''
This one is going to use the RNN features as global glimps for the story.
'''

import random
import numpy as np
import lmdb
import caffe

import theano
#import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
from theano import tensor as T, function, printing

import lasagne
from lasagne import layers
from lasagne import nonlinearities
import cPickle as pickle

import utils
import nn_utils

import os

import copy
import h5py
import pdb

floatX = theano.config.floatX

# For logging.
import climate
logging = climate.get_logger(__name__)
climate.enable_default_logging()

class DMN_batch:
    
    def __init__(self, data_dir, word2vec, word_vector_size, truncate_gradient, learning_rate, dim, cnn_dim, cnn_dim_fc, story_len,
                patches, mode, answer_module, memory_hops, batch_size, l2,
                normalize_attention, batch_norm, dropout, **kwargs):
        
        print "==> not used params in DMN class:", kwargs.keys()

        self.data_dir = data_dir
        self.learning_rate = learning_rate
        
        self.truncate_gradient = truncate_gradient
        self.word2vec = word2vec
        self.word_vector_size = word_vector_size
        self.dim = dim
        self.cnn_dim = cnn_dim
        self.cnn_dim_fc = cnn_dim_fc
        self.story_len = story_len
        self.mode = mode
        self.patches = patches
        self.answer_module = answer_module
        self.memory_hops = memory_hops
        self.batch_size = batch_size
        self.l2 = l2
        self.normalize_attention = normalize_attention
        self.batch_norm = batch_norm
        self.dropout = dropout

        self.vocab, self.ivocab = self._load_vocab(self.data_dir)

        self.train_story = None
        self.test_story = None
        self.train_dict_story, self.train_lmdb_env_fc, self.train_lmdb_env_conv = self._process_input_sind(self.data_dir, 'train')
        self.test_dict_story, self.test_lmdb_env_fc, self.test_lmdb_env_conv = self._process_input_sind(self.data_dir, 'val')

        self.train_story = self.train_dict_story.keys()
        self.test_story = self.test_dict_story.keys()
        self.vocab_size = len(self.vocab)

        # Since this is pretty expensive, we will pass a story each time.
        # We assume that the input has been processed such that the sequences of patches 
        # are snake like path.

        self.input_var = T.tensor4('input_var') # (batch_size, seq_len, patches, cnn_dim)
        self.q_var = T.matrix('q_var') # Now, it's a batch * image_sieze.
        self.answer_var = T.imatrix('answer_var') # answer of example in minibatch
        self.answer_mask = T.matrix('answer_mask')
        self.answer_inp_var = T.tensor3('answer_inp_var') # answer of example in minibatch
        
        print "==> building input module"
        self.W_inp_emb_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim))
        #self.b_inp_emb_in = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        # First, we embed the visual features before sending it to the bi-GRUs.

        inp_rhp = T.reshape(self.input_var, (self.batch_size* self.story_len* self.patches, self.cnn_dim))
        inp_rhp_dimshuffled = inp_rhp.dimshuffle(1,0)
        inp_rhp_emb = T.dot(self.W_inp_emb_in, inp_rhp_dimshuffled)
        inp_rhp_emb_dimshuffled = inp_rhp_emb.dimshuffle(1,0)
        inp_emb_raw = T.reshape(inp_rhp_emb_dimshuffled, (self.batch_size, self.story_len, self.patches, self.cnn_dim))
        inp_emb = T.tanh(inp_emb_raw) # Just follow the paper DMN for visual and textual QA.


        # Now, we use a bi-directional GRU to produce the input.
        # Forward GRU.
        self.inp_dim = self.dim/2 # since we have forward and backward
        self.W_inpf_res_in = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.cnn_dim))
        self.W_inpf_res_hid = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.inp_dim))
        self.b_inpf_res = nn_utils.constant_param(value=0.0, shape=(self.inp_dim,))
        
        self.W_inpf_upd_in = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.cnn_dim))
        self.W_inpf_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.inp_dim))
        self.b_inpf_upd = nn_utils.constant_param(value=0.0, shape=(self.inp_dim,))
        
        self.W_inpf_hid_in = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.cnn_dim))
        self.W_inpf_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.inp_dim))
        self.b_inpf_hid = nn_utils.constant_param(value=0.0, shape=(self.inp_dim,))
        # Backward GRU.
        self.W_inpb_res_in = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.cnn_dim))
        self.W_inpb_res_hid = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.inp_dim))
        self.b_inpb_res = nn_utils.constant_param(value=0.0, shape=(self.inp_dim,))
        
        self.W_inpb_upd_in = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.cnn_dim))
        self.W_inpb_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.inp_dim))
        self.b_inpb_upd = nn_utils.constant_param(value=0.0, shape=(self.inp_dim,))
        
        self.W_inpb_hid_in = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.cnn_dim))
        self.W_inpb_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.inp_dim, self.inp_dim))
        self.b_inpb_hid = nn_utils.constant_param(value=0.0, shape=(self.inp_dim,))

        # Now, we use the GRU to build the inputs.
        # Two-level of nested scan is unnecessary. It will become too complicated. Just use this one.
        inp_dummy = theano.shared(np.zeros((self.inp_dim, self.story_len), dtype = floatX))
        for i in range(self.batch_size):
            if i == 0:
                inp_1st_f, _ = theano.scan(fn = self.input_gru_step_forward,
                                    sequences = inp_emb[i,:].dimshuffle(1,2,0),
                                    outputs_info=T.zeros_like(inp_dummy), 
                                    truncate_gradient = self.truncate_gradient
                                    )

                inp_1st_b, _ = theano.scan(fn = self.input_gru_step_backward,
                                    sequences = inp_emb[i,:,::-1,:].dimshuffle(1,2,0),
                                    outputs_info=T.zeros_like(inp_dummy),
                                    truncate_gradient = self.truncate_gradient
                                    )
                # Now, combine them.
                inp_1st = T.concatenate([inp_1st_f.dimshuffle(2,0,1), inp_1st_b.dimshuffle(2,0,1)], axis = -1)
                self.inp_c = inp_1st.dimshuffle('x', 0, 1, 2)
            else:
                inp_f, _ = theano.scan(fn = self.input_gru_step_forward,
                                    sequences = inp_emb[i,:].dimshuffle(1,2,0),
                                    outputs_info=T.zeros_like(inp_dummy),
                                    truncate_gradient = self.truncate_gradient
                                    )

                inp_b, _ = theano.scan(fn = self.input_gru_step_backward,
                                    sequences = inp_emb[i,:,::-1,:].dimshuffle(1,2,0),
                                    outputs_info=T.zeros_like(inp_dummy),
                                    truncate_gradient = self.truncate_gradient
                                    )
                # Now, combine them.
                inp_fb = T.concatenate([inp_f.dimshuffle(2,0,1), inp_b.dimshuffle(2,0,1)], axis = -1)
                self.inp_c = T.concatenate([self.inp_c, inp_fb.dimshuffle('x', 0, 1, 2)], axis = 0)
        # Done, now self.inp_c should be batch_size x story_len x patches x cnn_dim
        # Eventually, we can flattern them.
        # Now, the input dimension is 1024 because we have forward and backward.
        inp_c_t = T.reshape(self.inp_c, (self.batch_size, self.story_len * self.patches, self.dim))
        inp_c_t_dimshuffled = inp_c_t.dimshuffle(0,'x', 1, 2)
        inp_batch = T.repeat(inp_c_t_dimshuffled, self.story_len, axis = 1)
        # Now, its ready for all the 5 images in the same story.
        # 50 * 980 * 512 
        self.inp_batch = T.reshape(inp_batch, (inp_batch.shape[0] * inp_batch.shape[1], inp_batch.shape[2], inp_batch.shape[3]))
        self.inp_batch_dimshuffled = self.inp_batch.dimshuffle(1,2,0) # 980 x 512 x 50
        
        
        # It's very simple now, the input module just need to map from cnn_dim to dim.
        logging.info('self.cnn_dim = %d', self.cnn_dim)

        print "==> building question module"
        # First is for the global glimpse.

        q_var_3 = T.reshape(self.q_var, (self.batch_size, self.story_len, self.cnn_dim_fc))

        q_var_shuffled = q_var_3.dimshuffle(1,2,0) # now: story_len * image_size * batch_size

        # This is the RNN used to produce the Global Glimpse
        self.W_qf_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim_fc))
        self.W_qf_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_qf_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_qf_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim_fc))
        self.W_qf_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_qf_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_qf_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim_fc))
        self.W_qf_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_qf_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        inp_dummy = theano.shared(np.zeros((self.dim, self.batch_size), dtype = floatX))

        q_glb,_ = theano.scan(fn = self.q_gru_step_forward, 
                                    sequences = q_var_shuffled,
                                    outputs_info = [T.zeros_like(inp_dummy)],
                                    truncate_gradient = self.truncate_gradient
                                    )
        q_glb_shuffled = q_glb.dimshuffle(2,0,1) # batch_size * seq_len * dim
        q_glb_last = q_glb_shuffled[:,-1,:] # batch_size * dim

        # Now, we also need to add the global glimpse, thus we need to use the rnn to build the attention glimpose.
        # Now, share the parameter with the input module.
        self.W_inp_emb_q = nn_utils.normal_param(std = 0.1, shape=(self.dim, self.cnn_dim_fc))
        self.b_inp_emb_q = nn_utils.normal_param(std = 0.1, shape=(self.dim,))
        q_var_shuffled = self.q_var.dimshuffle(1,0)

        inp_q = T.dot(self.W_inp_emb_q, q_var_shuffled) + self.b_inp_emb_q.dimshuffle(0,'x') # 512 x 50
        self.q_q = T.tanh(inp_q) # Since this is used to initialize the memory, we need to make it tanh.
        
        print "==> creating parameters for memory module"
        self.W_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        #self.W_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 0))
        self.W_2 = nn_utils.normal_param(std=0.1, shape=(1, self.dim))
        self.b_1 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.b_2 = nn_utils.constant_param(value=0.0, shape=(1,))
        

        print "==> building episodic memory module (fixed number of steps: %d)" % self.memory_hops
        memory = [self.q_q.copy()]
        for iter in range(1, self.memory_hops + 1):
            #m = printing.Print('mem')(memory[iter-1])
            current_episode = self.new_episode(memory[iter - 1])
            #current_episode = self.new_episode(m)
            #current_episode = printing.Print('current_episode')(current_episode)
            memory.append(self.GRU_update(memory[iter - 1], current_episode,
                                          self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                                          self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                                          self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid))                         
        
        last_mem_raw = memory[-1].dimshuffle((1, 0))
        
        net = layers.InputLayer(shape=(self.batch_size * self.story_len, self.dim), input_var=last_mem_raw)

        if self.batch_norm:
            net = layers.BatchNormLayer(incoming=net)
        if self.dropout > 0 and self.mode == 'train':
            net = layers.DropoutLayer(net, p=self.dropout)
        last_mem = layers.get_output(net).dimshuffle((1, 0))

        logging.info('last_mem size')
        print last_mem.shape.eval({self.input_var: np.random.rand(10,5,196,512).astype('float32'),
            self.q_var: np.random.rand(50, 4096).astype('float32')})
       
        print "==> building answer module"

        answer_inp_var_shuffled = self.answer_inp_var.dimshuffle(1,2,0)
        # Sounds good. Now, we need to map last_mem to a new space. 
        self.W_mem_emb = nn_utils.normal_param(std = 0.1, shape = (self.dim, self.dim * 3))
        self.W_inp_emb = nn_utils.normal_param(std = 0.1, shape = (self.dim, self.vocab_size + 1))

        def _dot2(x, W):
            return  T.dot(W, x)

        answer_inp_var_shuffled_emb,_ = theano.scan(fn = _dot2, sequences = answer_inp_var_shuffled,
                non_sequences = self.W_inp_emb,
                truncate_gradient = self.truncate_gradient
                ) # seq x dim x batch
        
        # Now, we also need to embed the image and use it to do the memory. 
        #q_q_shuffled = self.q_q.dimshuffle(1,0) # dim * batch.
        q_glb_dim = q_glb_last.dimshuffle(0,'x', 1) # batch_size * 1 * dim
        q_glb_repmat = T.repeat(q_glb_dim, self.story_len, 1) # batch_size * len * dim
        q_glb_rhp = T.reshape(q_glb_repmat, (q_glb_repmat.shape[0] * q_glb_repmat.shape[1], q_glb_repmat.shape[2]))

        init_ans = T.concatenate([self.q_q, last_mem, q_glb_rhp.dimshuffle(1, 0)], axis = 0)

        mem_ans = T.dot(self.W_mem_emb, init_ans) # dim x batchsize.

        mem_ans_dim = mem_ans.dimshuffle('x',0,1)

        answer_inp = T.concatenate([mem_ans_dim, answer_inp_var_shuffled_emb], axis = 0)
        
        # Now, we have both embedding. We can let them go to the rnn. 

        # We also need to map the input layer as well. 

        dummy = theano.shared(np.zeros((self.dim, self.batch_size * self.story_len), dtype=floatX))

        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size + 1, self.dim))
        
        self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        logging.info('answer_inp size')

        #print answer_inp.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
        #    self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32'),
        #    self.q_var: np.random.rand(10, 4096).astype('float32')})
        
        #last_mem = printing.Print('prob_sm')(last_mem)
        results, _ = theano.scan(fn = self.answer_gru_step,
                sequences = answer_inp,
                outputs_info = [ dummy ],
                truncate_gradient = self.truncate_gradient
                )
        # Assume there is a start token 
        #print results.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
        #    self.q_var: np.random.rand(10, 4096).astype('float32'), 
        #    self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')}, on_unused_input='ignore')
        results = results[1:-1,:,:] # get rid of the last token as well as the first one (image)
        #print results.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
        #    self.q_var: np.random.rand(10, 4096).astype('float32'), 
        #    self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')}, on_unused_input='ignore')
            
        # Now, we need to transform it to the probabilities.

        prob,_ = theano.scan(fn = lambda x, w: T.dot(w, x), sequences = results, non_sequences = self.W_a, 
                truncate_gradient = self.truncate_gradient
                )

        prob_shuffled = prob.dimshuffle(2,0,1) # b * len * vocab


        logging.info("prob shape.")
        #print prob.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
        #    self.q_var: np.random.rand(10, 4096).astype('float32'), 
        #    self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')})

        n = prob_shuffled.shape[0] * prob_shuffled.shape[1]
        prob_rhp = T.reshape(prob_shuffled, (n, prob_shuffled.shape[2]))
        prob_sm = nn_utils.softmax_(prob_rhp)
        self.prediction = prob_sm

        mask =  T.reshape(self.answer_mask, (n,))
        lbl = T.reshape(self.answer_var, (n,))

        self.params = [self.W_inp_emb_in, #self.b_inp_emb_in, 
                  self.W_inpf_res_in, self.W_inpf_res_hid,self.b_inpf_res,
                  self.W_inpf_upd_in, self.W_inpf_upd_hid, self.b_inpf_upd,
                  self.W_inpf_hid_in, self.W_inpf_hid_hid, self.b_inpf_hid,
                  self.W_inpb_res_in, self.W_inpb_res_hid, self.b_inpb_res,
                  self.W_inpb_upd_in, self.W_inpb_upd_hid, self.b_inpb_upd,
                  self.W_inpb_hid_in, self.W_inpb_hid_hid, self.b_inpb_hid,
                  self.W_qf_res_in, self.W_qf_res_hid, self.b_qf_res, 
                  self.W_qf_upd_in, self.W_qf_upd_hid, self.b_qf_upd,
                  self.W_qf_hid_in, self.W_qf_hid_hid, self.b_qf_hid,
                  self.W_inp_emb_q, self.b_inp_emb_q,
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid, #self.W_b
                  self.W_1, self.W_2, self.b_1, self.b_2, self.W_a,
                  self.W_mem_emb, self.W_inp_emb,
                  self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                  self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                  self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid,
                  ]            
                              
        print "==> building loss layer and computing updates"
        loss_vec = T.nnet.categorical_crossentropy(prob_sm, lbl)
        self.loss_ce = (mask * loss_vec ).sum() / mask.sum() 

        #self.loss_ce = T.nnet.categorical_crossentropy(results_rhp, lbl)
            
        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = self.loss_ce + self.loss_l2
            
        updates = lasagne.updates.adadelta(self.loss, self.params, learning_rate = self.learning_rate)
        #updates = lasagne.updates.momentum(self.loss, self.params, learning_rate=0.001)
        
        if self.mode == 'train':
            print "==> compiling train_fn"
            self.train_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.answer_mask, self.answer_inp_var], 
                                            outputs=[self.prediction, self.loss],
                                            updates=updates)
        
        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.answer_mask, self.answer_inp_var],
                                       outputs=[self.prediction, self.loss])
        
    
    
    def GRU_update(self, h, x, W_res_in, W_res_hid, b_res,
                         W_upd_in, W_upd_hid, b_upd,
                         W_hid_in, W_hid_hid, b_hid):
        """ mapping of our variables to symbols in DMN paper: 
        W_res_in = W^r
        W_res_hid = U^r
        b_res = b^r
        W_upd_in = W^z
        W_upd_hid = U^z
        b_upd = b^z
        W_hid_in = W
        W_hid_hid = U
        b_hid = b^h
        """
        z = T.nnet.sigmoid(T.dot(W_upd_in, x) + T.dot(W_upd_hid, h) + b_upd.dimshuffle(0, 'x'))
        r = T.nnet.sigmoid(T.dot(W_res_in, x) + T.dot(W_res_hid, h) + b_res.dimshuffle(0, 'x'))
        _h = T.tanh(T.dot(W_hid_in, x) + r * T.dot(W_hid_hid, h) + b_hid.dimshuffle(0, 'x'))
        return z * h + (1 - z) * _h
    
    
    def _empty_word_vector(self):
        return np.zeros((self.word_vector_size,), dtype=floatX)
    
    def _empty_inp_cnn_vector(self):
        return np.zeros((self.cnn_dim,), dtype=floatX)
    
    def input_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                                     self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                                     self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid)
    def input_gru_step_forward(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inpf_res_in, self.W_inpf_res_hid, self.b_inpf_res, 
                                     self.W_inpf_upd_in, self.W_inpf_upd_hid, self.b_inpf_upd,
                                     self.W_inpf_hid_in, self.W_inpf_hid_hid, self.b_inpf_hid)

    def q_gru_step_forward(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_qf_res_in, self.W_qf_res_hid, self.b_qf_res, 
                                     self.W_qf_upd_in, self.W_qf_upd_hid, self.b_qf_upd,
                                     self.W_qf_hid_in, self.W_qf_hid_hid, self.b_qf_hid)


    def input_gru_step_backward(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inpb_res_in, self.W_inpb_res_hid, self.b_inpb_res, 
                                     self.W_inpb_upd_in, self.W_inpb_upd_hid, self.b_inpb_upd,
                                     self.W_inpb_hid_in, self.W_inpb_hid_hid, self.b_inpb_hid)
   
    def answer_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                                     self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                                     self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)
    
        #y = nn_utils.softmax(T.dot(self.W_a, p))
        #return y
    
    def new_attention_step(self, ct, prev_g, mem, q_q):
        z = T.concatenate([ct, mem, q_q, ct * q_q, ct * mem, (ct - q_q) ** 2, (ct - mem) ** 2], axis=0)
        
        l_1 = T.dot(self.W_1, z) + self.b_1.dimshuffle(0, 'x')
        l_1 = T.tanh(l_1)
        l_2 = T.dot(self.W_2, l_1) + self.b_2.dimshuffle(0, 'x')
        G = T.nnet.sigmoid(l_2)[0]
        return G
        
        
    def new_episode_step(self, ct, g, prev_h):
        gru = self.GRU_update(prev_h, ct,
                             self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                             self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                             self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid)
        
        h = g * gru + (1 - g) * prev_h
        return h
     
       
    def new_episode(self, mem):
        #epi_dummy = theano.shared(np.zeros((self.dim,), dtype = floatX))
        g, g_updates = theano.scan(fn=self.new_attention_step,
            sequences=self.inp_batch_dimshuffled, #980 x 512 x 50
            non_sequences=[mem, self.q_q],
            #outputs_info=T.zeros_like(epi_dummy))
            outputs_info=T.zeros_like(self.inp_batch_dimshuffled[0][0]),
            truncate_gradient = self.truncate_gradient )
        
        if (self.normalize_attention):
            g = nn_utils.softmax(g)
        
        #epi_dummy2 = theano.shared(np.zeros((self.dim,self.dim), dtype = floatX))
        e, e_updates = theano.scan(fn=self.new_episode_step,
            sequences=[self.inp_batch_dimshuffled, g],
            #outputs_info=T.zeros_like(epi_dummy2))
            outputs_info=T.zeros_like(self.inp_batch_dimshuffled[0]),
            truncate_gradient = self.truncate_gradient )
        
        e_list = []
        for index in range(self.batch_size * self.story_len):
            e_list.append(e[-1, :, index])
        return T.stack(e_list).dimshuffle((1, 0))
   
   
    def save_params(self, file_name, epoch, **kwargs):
        with open(file_name, 'w') as save_file:
            pickle.dump(
                obj = {
                    'params' : [x.get_value() for x in self.params],
                    'epoch' : epoch, 
                    'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
                },
                file = save_file,
                protocol = -1
            )
    
    
    def load_state(self, file_name):
        print "==> loading state %s" % file_name
        with open(file_name, 'r') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)

    def _process_batch_sind(self, batch_index, split = 'train'):
        # Now, randomly select one story.

        start_index = self.batch_size * batch_index

        split_story = None
        if split == 'train':
            split_lmdb_env_fc = self.train_lmdb_env_fc
            split_lmdb_env_conv = self.train_lmdb_env_conv
            split_story = self.train_story
            split_dict_story = self.train_dict_story
        else:
            split_lmdb_env_fc = self.test_lmdb_env_fc
            split_lmdb_env_conv = self.test_lmdb_env_conv
            split_story = self.test_story
            split_dict_story = self.test_dict_story

        # make sure it's small than the number of stories.
        start_index = start_index % len(split_story)
        # make sure there is enough for a batch.
        start_index = min(start_index, len(split_story) - self.batch_size)
        # Now, we select the stories.
        stories = split_story[start_index:start_index+self.batch_size]
        #    slids.append( random.choice(range(len(split_dict_story[sid]))))

        max_inp_len = 0
        max_q_len = 1 # just be 1.
        max_ans_len = 0
        for sid in stories:
            max_inp_len = max(max_inp_len, len(split_dict_story[sid])-1)
            #max_q_len = max(max_q_len, split_dict_story[sid][slid][1])
            for slid in split_dict_story[sid]:
                max_ans_len = max(max_ans_len, len(slid[-1][-1]))
        
        max_ans_len += 1 # this is for the start token.
        # in our case, it is pretty similar to the word-level dmn,

        questions = []
        # batch x story_len x fea
        inputs = []
        answers = []
        answers_inp = []
        answers_mask = []
        max_key_len = 12

        with split_lmdb_env_fc.begin() as txn_fc:
            with split_lmdb_env_conv.begin() as txn_conv:
                for sid in stories:
                    inp = [] # story_len x patches x fea.
                    anno = split_dict_story[sid]
                    question = []
                    answer = []
                    answer_mask = []
                    answer_inp = []

                    for slid in split_dict_story[sid]:
                        input_anno = slid
                        img_id = input_anno[1][0]
                        while len(img_id) < max_key_len:
                            img_id = '0' + img_id

                        fc_raw = txn_fc.get(img_id.encode('ascii'))
                        fc_fea = caffe.proto.caffe_pb2.Datum()
                        fc_fea.ParseFromString(fc_raw)
                        question.append( np.fromstring(fc_fea.data, dtype = np.float32))
                        # Now, it is the inputs, we can use the other images other than current one.
                        conv_raw = txn_conv.get(img_id.encode('ascii'))
                        conv_datum = caffe.proto.caffe_pb2.Datum()
                        conv_datum.ParseFromString(conv_raw)
                        conv_fea = np.fromstring(conv_datum.data, dtype = np.float32)
                        x = conv_fea.reshape(conv_datum.channels, conv_datum.height, conv_datum.width) # 512 x 14 x 14
                        x = x.reshape(conv_datum.channels, conv_datum.height * conv_datum.width)
                        x = x.swapaxes(0,1)
                        inp.append(x)
                        #now for answer.

                        a = []
                        a.append(self.vocab_size) # start token.
                        a.extend(input_anno[1][2]) # this is the index for the captions.
                        a_inp = np.zeros((max_ans_len, self.vocab_size + 1), dtype = floatX)
                        a_mask = []
                        for ans_idx, w_idx in enumerate(a):
                            a_inp[ans_idx, w_idx] = 1

                        a_mask = [ 1 for i in range(len(a) -1) ]
                        while len(a) < max_ans_len: # this does not matter.
                            a.append( -1 )
                            a_mask.append(0)

                        a = a[1:]
                        answer.append(np.array(a).astype(np.int32))
                        answer_mask.append(np.array(a_mask).astype(np.int32))
                        answer_inp.append(np.array(a_inp).astype(floatX))
                    
                    #pdb.set_trace()
                    question = np.stack(question, axis = 0)
                    questions.append(question)
                    inp = np.stack(inp, axis = 0) # #story_len x patches x fea
                    inputs.append(inp)
                    answer = np.stack(answer, axis = 0) # story_len x max_answer_len
                    answers.append(answer)
                    answer_mask = np.stack(answer_mask, axis =0) # story_len x max_answer_len -1
                    answers_mask.append(answer_mask)
                    answer_inp = np.stack(answer_inp, axis = 0) # story_len x max_answer_len
                    #pdb.set_trace()
                    answers_inp.append(answer_inp)

        # Finally, we transform them into numpy array.
        inputs = np.stack(inputs, axis = 0)
        inputs = np.array(inputs).astype(floatX)
        #questions = np.array(questions).astype(floatX)
        questions = np.stack(questions, axis = 0)
        questions = np.array(questions).astype(floatX)
        answers = np.array(answers).astype(np.int32)
        answers_mask = np.array(answers_mask).astype(floatX)
        #print answers_mask
        answers_inp = np.stack(answers_inp, axis = 0)
        questions = np.reshape(questions, (questions.shape[0] * questions.shape[1], questions.shape[2]))
        answers = np.reshape(answers, (answers.shape[0] * answers.shape[1], answers.shape[2]))
        answers_inp = np.reshape(answers_inp, (answers_inp.shape[0] * answers_inp.shape[1], answers_inp.shape[2], answers_inp.shape[3]))
        answers_mask = np.reshape(answers_mask, (answers_mask.shape[0] * answers_mask.shape[1], answers_mask.shape[2]))

        #print inputs.shape
        #print questions.shape
        #print answers.shape
        #print answers_inp.shape
        #print answers_mask.shape


        return inputs, questions, answers, answers_inp, answers_mask
   
    
    def _process_batch(self, _inputs, _questions, _answers, _fact_counts, _input_masks):
        inputs = copy.deepcopy(_inputs)
        questions = copy.deepcopy(_questions)
        answers = copy.deepcopy(_answers)
        fact_counts = copy.deepcopy(_fact_counts)
        input_masks = copy.deepcopy(_input_masks)
        
        zipped = zip(inputs, questions, answers, fact_counts, input_masks)
        
        max_inp_len = 0
        max_q_len = 0
        max_fact_count = 0
        for inp, q, ans, fact_count, input_mask in zipped:
            max_inp_len = max(max_inp_len, len(inp))
            max_q_len = max(max_q_len, len(q))
            max_fact_count = max(max_fact_count, fact_count)
        
        questions = []
        inputs = []
        answers = []
        fact_counts = []
        input_masks = []
        
        for inp, q, ans, fact_count, input_mask in zipped:
            while(len(inp) < max_inp_len):
                inp.append(self._empty_word_vector())
            
            while(len(q) < max_q_len):
                q.append(self._empty_word_vector())
    
            while(len(input_mask) < max_fact_count):
                input_mask.append(-1)
            
            inputs.append(inp)
            questions.append(q)
            answers.append(ans)
            fact_counts.append(fact_count)
            input_masks.append(input_mask)
            
        inputs = np.array(inputs).astype(floatX)
        questions = np.array(questions).astype(floatX)
        answers = np.array(answers).astype(np.int32)
        fact_counts = np.array(fact_counts).astype(np.int32)
        input_masks = np.array(input_masks).astype(np.int32)

        return inputs, questions, answers, fact_counts, input_masks 
    
    def _process_input_sind(self, data_dir, split = 'train'):

        # Some lmdb configuration.
        lmdb_dir_fc =  os.path.join(self.data_dir, split, 'fea_vgg16_fc7_lmdb_lmdb')
        lmdb_dir_conv = os.path.join(self.data_dir, split, 'imgs_resized_vgg16_conv5_3_lmdb_lmdb')

        lmdb_env_fc = lmdb.open(lmdb_dir_fc, readonly = True)
        lmdb_env_conv = lmdb.open(lmdb_dir_conv, readonly = True)

        split_dir = os.path.join(data_dir, split)
        anno_fn = os.path.join(split_dir,'annotions_filtered.txt')

        # Now load the stories.
        dict_story = {}
        with open(anno_fn ,'r') as fid:
            for aline in fid:
                parts = aline.strip().split()
                flickr_id = parts[0]
                sid = int(parts[2])
                slid = int(parts[3])
                if sid not in dict_story:
                    dict_story[sid] = {}
                dict_story[sid][slid] = []
                dict_story[sid][slid].append(flickr_id)
                inp_v = []
                #inp_v = [ utils.process_word2(word = w,
                #                        word2vec = self.word2vec,
                #                        vocab = self.vocab,
                #                        word_vector_size = self.word_vector_size,
                #                        to_return = 'word2vec') for w in parts[4:] ]

                inp_y = [ utils.process_word2(word = w,
                                        word2vec = self.word2vec,
                                        vocab = self.vocab,
                                        word_vector_size = self.word_vector_size,
                                        to_return = 'index', silent=True) for w in parts[4:] ]


                dict_story[sid][slid].append( inp_v )
                dict_story[sid][slid].append( inp_y )

        # Just in case, we sort all the stories in line.
        for sid in dict_story:
            story = dict_story[sid].items()
            sorted(story, key = lambda x: x[0])
            dict_story[sid] = story

        return dict_story, lmdb_env_fc, lmdb_env_conv

    def _load_vocab(self, data_dir):
        v_fn = os.path.join(data_dir, 'vocab_fixed_glove.txt')
        vocab = {}
        ivocab = {}
        with open(v_fn, 'r') as fid:
            for aline in fid:
                parts = aline.strip().split()
                vocab[parts[0]] = len(vocab)
                ivocab[len(ivocab)] = parts[0]
        # Now, add UNK
        vocab['UNK'] = len(vocab)
        ivocab[len(ivocab)] = 'UNK'
        vocab['[male]'] = len(vocab)
        ivocab[len(ivocab)] = '[male]'
        vocab['[female]'] = len(vocab)
        ivocab[len(ivocab)] = '[female]'

        logging.info('len(vocab) / len(ivocab) = %d/%d', len(vocab), len(ivocab))
        return vocab, ivocab
 
    #def _load_vocab(self, data_dir):
    #    v_fn = os.path.join(data_dir, 'vocab.txt')
    #    vocab = {}
    #    ivocab = {}
    #    with open(v_fn, 'r') as fid:
    #        for aline in fid:
    #            parts = aline.strip().split()
    #            vocab[parts[1]] = int(parts[0])
    #            ivocab[int(parts[0])] = parts[1]
    #    return vocab, ivocab
    
    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            cnt = 0
            #for story in self.train_dict_story:
            #    cnt += len(self.train_dict_story[story])
            cnt = len(self.train_dict_story)

            return cnt / self.batch_size
        elif (mode == 'test'):
            cnt = 0
            for story in self.test_dict_story:
                cnt += len(self.test_dict_story[story])
            return cnt / self.batch_size
        else:
            raise Exception("unknown mode")
    
    def shuffle_train_set(self):
        if self.train_story:
            random.shuffle(self.train_story)
        else:
            self.train_story = self.train_dict_story.keys()

    def step(self, batch_index, mode):
        if mode == "train" and self.mode == "test":
            raise Exception("Cannot train during test mode")
        
        if mode == "train":
            theano_fn = self.train_fn 
        if mode == "test":    
            theano_fn = self.test_fn 
        
        inp, q, ans, ans_inp, ans_mask = self._process_batch_sind(batch_index, mode)
        
        ret = theano_fn(inp, q, ans, ans_mask, ans_inp)
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"prediction": ret[0],
                "answers": ans,
                "current_loss": ret[1],
                "skipped": 0,
                "log": "pn: %.3f" % param_norm,
                }
        
