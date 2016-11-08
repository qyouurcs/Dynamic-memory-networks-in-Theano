'''
Delete the RNN on the local patches.

1. Use all the images in the same story to build the initial state to the rnn.
2. Use each image's locla patches to attend.

'''
import random
import numpy as np
import lmdb
import caffe

import theano
from theano.compile.nanguardmode import NanGuardMode
from theano import tensor as T, function, printing
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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

import sys
reload(sys)  
sys.setdefaultencoding('utf8')

class DMN_batch:
    
    def __init__(self, data_dir, word2vec, word_vector_size, dim, cnn_dim, story_len,
                patches,cnn_dim_fc,truncate_gradient, learning_rate,
                mode, answer_module, memory_hops, batch_size, l2,
                normalize_attention, batch_norm, dropout, **kwargs):
        
        print "==> not used params in DMN class:", kwargs.keys()

        self.data_dir = data_dir
        self.truncate_gradient = truncate_gradient
        self.learning_rate = learning_rate

        self.trng = RandomStreams(1234)
        
        self.word2vec = word2vec
        self.word_vector_size = word_vector_size
        self.dim = dim
        self.cnn_dim = cnn_dim
        self.cnn_dim_fc = cnn_dim_fc
        self.story_len = story_len
        self.patches = patches
        self.mode = mode
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
        self.train_dict_story, self.train_lmdb_env_fc, self.train_lmdb_env_conv = self._process_input_sind_lmdb(self.data_dir, 'train')
        self.test_dict_story, self.test_lmdb_env_fc, self.test_lmdb_env_conv = self._process_input_sind_lmdb(self.data_dir, 'val')

        self.train_story = self.train_dict_story.keys()
        self.test_story = self.test_dict_story.keys()
        self.vocab_size = len(self.vocab)
        self.alpha_entropy_c = 0.02 # for hard attention.
        
        # This is the local patch of each image.
        self.input_var = T.tensor4('input_var') # (batch_size, seq_len, patches, cnn_dim)
        self.q_var = T.tensor3('q_var') # Now, it's a batch * story_len * image_sieze.
        self.answer_var = T.ivector('answer_var') # answer of example in minibatch
        self.answer_mask = T.matrix('answer_mask')
        self.answer_idx = T.imatrix('answer_idx') # batch x seq
        self.answer_inp_var = T.tensor3('answer_inp_var') # answer of example in minibatch

        self.init_h = T.matrix('init_h') # This is only used for the prediction function.
        self.story_idx = T.iscalar('story_idx')
        
        print "==> building input module"
        # It's very simple now, the input module just need to map from cnn_dim to dim.
        logging.info('self.cnn_dim = %d', self.cnn_dim)
        logging.info('self.cnn_dim_fc = %d', self.cnn_dim_fc)
        logging.info('self.dim = %d', self.dim)
        self.W_q_emb_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim_fc))
        self.b_q_emb_in = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        logging.info('Building the glob attention model')
        self.W_glb_att_1 = nn_utils.normal_param(std = 0.1, shape = (self.dim, 2 * self.dim))
        self.W_glb_att_2 = nn_utils.normal_param(std = 0.1, shape = (1, self.dim))
        self.b_glb_att_1 = nn_utils.constant_param(value = 0.0, shape = (self.dim,))
        self.b_glb_att_2 = nn_utils.constant_param(value = 0.0, shape = (1,))

        q_var_shuffled = self.q_var.dimshuffle(1,2,0) # seq x cnn x batch.

        def _dot(x, W, b):
            return T.tanh( T.dot(W, x) + b.dimshuffle(0, 'x'))

        q_var_shuffled_emb,_ = theano.scan(fn = _dot, sequences= q_var_shuffled, non_sequences = [self.W_q_emb_in, self.b_q_emb_in])
        #print 'q_var_shuffled_emb', q_var_shuffled_emb.shape.eval({self.q_var:np.random.rand(2,5,4096).astype('float32')})
        q_var_emb =  q_var_shuffled_emb.dimshuffle(2,0,1) # batch x seq x emb_size
        q_var_emb_ext = q_var_emb.dimshuffle(0,'x',1,2)
        q_var_emb_ext = T.repeat(q_var_emb_ext, q_var_emb.shape[1],1) # batch x seq x seq x emb_size
        q_var_emb_rhp = T.reshape( q_var_emb, (q_var_emb.shape[0] * q_var_emb.shape[1], q_var_emb.shape[2]))
        q_var_emb_ext_rhp = T.reshape(q_var_emb_ext, (q_var_emb_ext.shape[0] * q_var_emb_ext.shape[1],q_var_emb_ext.shape[2], q_var_emb_ext.shape[3]))
        q_var_emb_ext_rhp = q_var_emb_ext_rhp.dimshuffle(0,2,1)
        q_idx = T.arange(self.story_len).dimshuffle('x',0)
        q_idx = T.repeat(q_idx,self.batch_size, axis = 0)
        q_idx = T.reshape(q_idx, (q_idx.shape[0]* q_idx.shape[1],))

        alpha,_ = theano.scan(fn = self.new_attention_step_glob, sequences = [q_var_emb_rhp, q_var_emb_ext_rhp, q_idx] )

        att_alpha = alpha[1]
        att_alpha_a = alpha[0]

        self.W_inp_emb_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim))
        self.b_inp_emb_in = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        inp_rhp = T.reshape(self.input_var, (self.batch_size* self.story_len* self.patches, self.cnn_dim))
        inp_rhp_dimshuffled = inp_rhp.dimshuffle(1,0)
        inp_rhp_emb = T.dot(self.W_inp_emb_in, inp_rhp_dimshuffled) + self.b_inp_emb_in.dimshuffle(0,'x')
        inp_rhp_emb_dimshuffled = inp_rhp_emb.dimshuffle(1,0)
        inp_emb_raw = T.reshape(inp_rhp_emb_dimshuffled, (self.batch_size, self.story_len, self.patches, self.cnn_dim))
        inp_emb = T.tanh(inp_emb_raw) # Just follow the paper DMN for visual and textual QA.

        att_alpha_sample = self.trng.multinomial(pvals = att_alpha, dtype=theano.config.floatX)
        att_mask = att_alpha_sample.argmax(1)

        idx_t = T.repeat(T.arange(self.input_var.shape[0]), self.input_var.shape[1])
        att_input =  inp_emb[idx_t, att_mask,:,:] # (batch x seq) x batches x emb_size
        att_input = T.reshape(att_input, (self.batch_size, self.story_len, self.patches, self.dim))
        
        self.inp_c = T.reshape(att_input, (att_input.shape[0] * att_input.shape[1], att_input.shape[2], att_input.shape[3]))
        self.inp_c = self.inp_c.dimshuffle(1,2,0)

        logging.info('building question module')
        self.W_qf_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_qf_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_qf_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_qf_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_qf_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_qf_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_qf_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_qf_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_qf_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        inp_dummy = theano.shared(np.zeros((self.dim, self.batch_size), dtype = floatX))

        q_var_shuffled_emb_reversed = q_var_shuffled_emb[::-1,:,:] # seq x emb_size x batch
        q_glb,_ = theano.scan(fn = self.q_gru_step_forward, 
                                    sequences = q_var_shuffled_emb_reversed,
                                    outputs_info = [T.zeros_like(inp_dummy)])
        q_glb_shuffled = q_glb.dimshuffle(2,0,1) # batch_size * seq_len * dim
        q_glb_last = q_glb_shuffled[:,-1,:] # batch_size * dim

        q_net = layers.InputLayer(shape=(self.batch_size*self.story_len, self.dim), input_var=q_var_emb_rhp)
        if self.batch_norm:
            q_net = layers.BatchNormLayer(incoming=q_net)
        if self.dropout > 0 and self.mode == 'train':
            q_net = layers.DropoutLayer(q_net, p=self.dropout)
        self.q_q = layers.get_output(q_net).dimshuffle(1,0)

        #print "==> creating parameters for memory module"
        logging.info('creating parameters for memory module')
        self.W_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_mem_update1 = nn_utils.normal_param(std=0.1, shape=(self.dim , self.dim* 2))
        self.b_mem_upd1 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.W_mem_update2 = nn_utils.normal_param(std=0.1, shape=(self.dim,self.dim*2))
        self.b_mem_upd2 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.W_mem_update3 = nn_utils.normal_param(std=0.1, shape=(self.dim , self.dim*2))
        self.b_mem_upd3 = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        self.W_mem_update = [self.W_mem_update1,self.W_mem_update2,self.W_mem_update3]
        self.b_mem_update = [self.b_mem_upd1,self.b_mem_upd2, self.b_mem_upd3]
        
        self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 0))
        self.W_2 = nn_utils.normal_param(std=0.1, shape=(1, self.dim))
        self.b_1 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.b_2 = nn_utils.constant_param(value=0.0, shape=(1,))
        logging.info('==> building episodic memory module (fixed number of steps: %d)', self.memory_hops)
        memory = [self.q_q.copy()]
        for iter in range(1, self.memory_hops + 1):
            #m = printing.Print('mem')(memory[iter-1])
            current_episode = self.new_episode(memory[iter - 1])
            # Replace GRU with ReLU activation + MLP.
            c = T.concatenate([memory[iter - 1], current_episode], axis = 0)
            cur_mem = T.dot(self.W_mem_update[iter-1], c) + self.b_mem_update[iter-1].dimshuffle(0,'x')
            memory.append(T.nnet.relu(cur_mem))
        
        last_mem_raw = memory[-1].dimshuffle((1, 0))
        
        net = layers.InputLayer(shape=(self.batch_size * self.story_len, self.dim), input_var=last_mem_raw)

        if self.batch_norm:
            net = layers.BatchNormLayer(incoming=net)
        if self.dropout > 0 and self.mode == 'train':
            net = layers.DropoutLayer(net, p=self.dropout)
        last_mem = layers.get_output(net).dimshuffle((1, 0))

        print "==> building answer module"

        answer_inp_var_shuffled = self.answer_inp_var.dimshuffle(1,2,0)
        # Sounds good. Now, we need to map last_mem to a new space. 
        self.W_mem_emb = nn_utils.normal_param(std = 0.1, shape = (self.dim, self.dim * 2))
        self.b_mem_emb = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.W_inp_emb = nn_utils.normal_param(std = 0.1, shape = (self.dim, self.word_vector_size))
        self.b_inp_emb = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        def _dot2(x, W, b):
            #return  T.tanh(T.dot(W, x) + b.dimshuffle(0,'x'))
            return  T.dot(W, x) + b.dimshuffle(0,'x')

        answer_inp_var_shuffled_emb,_ = theano.scan(fn = _dot2, sequences = answer_inp_var_shuffled,
                non_sequences = [self.W_inp_emb,self.b_inp_emb] ) # seq x dim x batch
        
        init_ans = T.concatenate([self.q_q, last_mem], axis = 0) # dim x (batch x self.story_len)

        mem_ans = T.dot(self.W_mem_emb, init_ans) + self.b_mem_emb.dimshuffle(0,'x') # dim x (batchsize x self.story_len)
        #mem_ans_dim = mem_ans.dimshuffle('x',0,1)
        mem_ans_rhp = T.reshape(mem_ans.dimshuffle(1,0), (self.batch_size, self.story_len, mem_ans.shape[0]))
        mem_ans_dim = mem_ans_rhp.dimshuffle(1,2,0)
        answer_inp = answer_inp_var_shuffled_emb
        #answer_inp = T.concatenate([mem_ans_dim, answer_inp_var_shuffled_emb], axis = 0) #seq + 1 x dim x (batch-size x self.story+len)
        # Now, each answer got its input, our next step is to obtain the sequences.
        answer_inp_shu = answer_inp.dimshuffle(2,0,1)
        answer_inp_shu_rhp = T.reshape(answer_inp_shu, (self.batch_size, self.story_len, answer_inp_shu.shape[1],\
                answer_inp_shu.shape[2]))

        answer_inp =  answer_inp_shu_rhp.dimshuffle(1,2,3,0) # story_len x seq + 1 x dim x batch_size


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

        self.W_ans_map = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim * 2))
        self.b_ans_map = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        results = None
        r = None

        dummy = theano.shared(np.zeros((self.dim, self.batch_size), dtype=floatX))
        for i in range(self.story_len):
            answer_inp_i = answer_inp[i,:] # seq + 1 x dim x batch_size
            mem_ans_dim_i  = mem_ans_dim[i,:] # dim x batch_size
            if i == 0:
                q_glb_inp = q_glb_last.dimshuffle('x', 1, 0) #1 x dim x batch_size
                answer_inp_i = T.concatenate([q_glb_inp, answer_inp_i], axis = 0)

                init_h = T.concatenate([dummy, mem_ans_dim_i], axis = 0)
                init_h = T.dot(self.W_ans_map, init_h) + self.b_ans_map.dimshuffle(0,'x')
                init_h = T.tanh(init_h)
                r, _ = theano.scan(fn = self.answer_gru_step,
                    sequences = answer_inp_i,
                    truncate_gradient = self.truncate_gradient,
                    outputs_info = [ init_h])
                r = r[1:,:] # get rid of the first glob one.
                results = r.dimshuffle('x', 0, 1, 2)
            else:
                prev_h = r[self.answer_idx[:,i],:,T.arange(self.batch_size)]
                h_ = T.concatenate([prev_h.dimshuffle(1,0), mem_ans_dim_i], axis = 0)
                h_ = T.dot(self.W_ans_map, h_) + self.b_ans_map.dimshuffle(0,'x')
                h_ = T.tanh(h_)

                r,_ = theano.scan(fn = self.answer_gru_step,
                        sequences = answer_inp_i,
                        truncate_gradient = self.truncate_gradient,
                        outputs_info = [ h_ ])
                results = T.concatenate([results, r.dimshuffle('x', 0, 1, 2)])
        ## results: story_len x seq+1 x dim x batch_size
        #########################################################################################
        # This is for the beam search.
        #########################################################################################
        beam_answer_inp = answer_inp[self.story_idx,:]
        beam_mem_ans_dim = mem_ans_dim[self.story_idx,:]
        beam_q_glb_inp = q_glb_last.dimshuffle('x', 1, 0)
        beam_anp_seq = theano.ifelse.ifelse( T.le(self.story_idx, 0), \
                T.concatenate([q_glb_inp, beam_answer_inp], axis = 0),\
                beam_answer_inp)
        beam_init_h = T.concatenate([ self.init_h, beam_mem_ans_dim], axis = 0) 
        beam_init_h = T.dot(self.W_ans_map, beam_init_h) + self.b_ans_map.dimshuffle(0, 'x')
        beam_init_h = T.tanh(beam_init_h)
        beam_r,_ = theano.scan(fn = self.answer_gru_step,
                sequences = beam_anp_seq,
                truncate_gradient = self.truncate_gradient,
                outputs_info = [ beam_init_h ] )
        
        beam_r = theano.ifelse.ifelse( T.le(self.story_idx, 0), \
                beam_r[1:,:,:],\
                beam_r)
        # Now, beam_r: seq+1 x dim x batch_size
        beam_prob, _ = theano.scan(fn = lambda x, w: T.dot(w, x), sequences = beam_r, non_sequences = self.W_a)
        beam_preds = beam_prob.dimshuffle(2,0,1) # b * len * vocab
        beam_preds_rhp =  T.reshape(beam_preds, (beam_preds.shape[0] * beam_preds.shape[1], beam_preds.shape[2]))
        beam_preds_rhp_sm = nn_utils.softmax(beam_preds_rhp)
        self.beam_pred = T.reshape(beam_preds_rhp_sm, (beam_preds.shape[0], beam_preds.shape[1], beam_preds.shape[2]))

        ###########################################################################################################

        results = results.dimshuffle(3,0,1,2)
        results = T.reshape(results, (self.batch_size * self.story_len, results.shape[2], results.shape[3]))
        results = results.dimshuffle(1,2,0) # seq_len x dim x (batch x seq)

        prob,_ = theano.scan(fn = lambda x, w: T.dot(w, x), sequences = results, non_sequences = self.W_a )
        preds = prob
        prob = prob[:-1,:,:]

        prob_shuffled = prob.dimshuffle(2,0,1) # b * len * vocab
        preds_shuffled = preds.dimshuffle(2,0,1)

        n = prob_shuffled.shape[0] * prob_shuffled.shape[1]
        n_preds = preds_shuffled.shape[0] * preds_shuffled.shape[1]

        prob_rhp = T.reshape(prob_shuffled, (n, prob_shuffled.shape[2]))
        preds_rhp = T.reshape(preds_shuffled, (n_preds, preds_shuffled.shape[2]))

        prob_sm = nn_utils.softmax_(prob_rhp)
        preds_sm = nn_utils.softmax_(preds_rhp)
        self.prediction = prob_sm # this one is for the training.

        self.pred = T.reshape(preds_sm, (preds_shuffled.shape[0], preds_shuffled.shape[1], preds_shuffled.shape[2]))

        mask =  T.reshape(self.answer_mask, (n,))
        lbl = T.reshape(self.answer_var, (n,))

        self.params = [
                self.W_q_emb_in, self.b_q_emb_in,
                self.W_glb_att_1, self.W_glb_att_2, self.b_glb_att_1, self.b_glb_att_2,
                self.W_inp_emb_in, self.b_inp_emb_in, 
                self.W_qf_res_in, self.W_qf_res_hid, self.b_qf_res,
                self.W_qf_upd_in, self.W_qf_upd_hid, self.b_qf_upd,
                self.W_qf_hid_in, self.W_qf_hid_hid, self.b_qf_hid,
                self.W_mem_emb, self.W_inp_emb,self.b_mem_emb, self.b_inp_emb,
                self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid, #self.W_b
                #self.W_mem_emb, self.W_inp_emb,self.b_mem_emb, self.b_inp_emb,
                self.W_1, self.W_2, self.b_1, self.b_2, self.W_a,
                self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid,
                self.W_ans_map, self.b_ans_map,
                ]
        self.params += self.W_mem_update
        self.params += self.b_mem_update
                              
                              
        print "==> building loss layer and computing updates"
        reward_prob = prob_sm[T.arange(n), lbl]
        reward_prob = T.reshape(reward_prob, (prob_shuffled.shape[0], prob_shuffled.shape[1]))

        loss_vec = T.nnet.categorical_crossentropy(prob_sm, lbl)
        self.loss_ce = (mask * loss_vec ).sum() / mask.sum() 

        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = self.loss_ce + self.loss_l2
        self.baseline_time = theano.shared(np.float32(0.), name='baseline_time')
        alpha_entropy_c = theano.shared(np.float32(self.alpha_entropy_c), name='alpha_entropy_c')
        mean_r = (self.answer_mask * reward_prob).sum(1) / self.answer_mask.sum(1) # or just fixed it as 1.
        mean_r = mean_r[0,None]
        grads = T.grad(self.loss, wrt=self.params,
                     disconnected_inputs='raise',
                     known_grads={att_alpha_a:(mean_r - self.baseline_time)*
                     (att_alpha_sample/(att_alpha_a + 1e-10)) + alpha_entropy_c*(T.log(att_alpha_a + 1e-10) + 1)})

            
        updates = lasagne.updates.adadelta(grads, self.params, learning_rate = self.learning_rate)
        updates[self.baseline_time] =  self.baseline_time * 0.9 + 0.1 * mean_r.mean()
        
        if self.mode == 'train':
            logging.info("compiling train_fn")
            self.train_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.answer_mask, self.answer_inp_var, self.answer_idx], 
                                            outputs=[self.prediction, self.loss],
                                            updates=updates)
        
        logging.info("compiling test_fn")
        self.test_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.answer_mask, self.answer_inp_var, self.answer_idx],
                                       outputs=[self.prediction, self.loss])
        
        logging.info("compiling pred_fn")
        self.pred_fn= theano.function(inputs=[self.input_var, self.q_var, self.answer_inp_var, self.answer_idx],
                                       outputs=[self.pred])
    
        logging.info("compiling beam pred_fn")
        self.beam_pred_fn= theano.function(inputs=[self.input_var, self.q_var, self.answer_inp_var, self.init_h, self.story_idx],
                                       outputs=[self.beam_pred, beam_r.dimshuffle(2,0,1)])

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

    def q_gru_step_forward(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_qf_res_in, self.W_qf_res_hid, self.b_qf_res, 
                                     self.W_qf_upd_in, self.W_qf_upd_hid, self.b_qf_upd,
                                     self.W_qf_hid_in, self.W_qf_hid_hid, self.b_qf_hid)

   
    def input_gru_step_forward(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inpf_res_in, self.W_inpf_res_hid, self.b_inpf_res, 
                                     self.W_inpf_upd_in, self.W_inpf_upd_hid, self.b_inpf_upd,
                                     self.W_inpf_hid_in, self.W_inpf_hid_hid, self.b_inpf_hid)
    def input_gru_step_backward(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inpb_res_in, self.W_inpb_res_hid, self.b_inpb_res, 
                                     self.W_inpb_upd_in, self.W_inpb_upd_hid, self.b_inpb_upd,
                                     self.W_inpb_hid_in, self.W_inpb_hid_hid, self.b_inpb_hid)

    def _empty_word_vector(self):
        return np.zeros((self.word_vector_size,), dtype=floatX)
    
    def _empty_inp_cnn_vector(self):
        return np.zeros((self.cnn_dim,), dtype=floatX)
    
    def input_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                                     self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                                     self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid)
    
    def answer_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                                     self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                                     self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)

        #y = nn_utils.softmax(T.dot(self.W_a, p))
        #return y

    def new_attention_step_glob(self, c_i, seq_i, i):
        # c_i is a vector, which is the embeding of the glob
        # seq_i is a matrix, which is the remaining of the seq imags. dim x seq 
        
        c_i_m = c_i.dimshuffle(0,'x')
        c_i_m_r = T.repeat( c_i_m, seq_i.shape[1], 1)

        t = T.concatenate([ c_i_m_r * seq_i, T.abs_( c_i_m_r - seq_i) ], axis = 0) # (2 * dim )  x seq 
        #print t.shape.eval({c_i:np.random.rand(512,).astype('float32'), seq_i:np.random.rand(512,5).astype('float32')})

        a_1 = T.dot(self.W_glb_att_1, t) + self.b_glb_att_1.dimshuffle(0,'x') # dim * seq - 1

        a_1 = T.tanh(a_1)
        a_2 = T.dot(self.W_glb_att_2, a_1) + self.b_glb_att_2.dimshuffle(0,'x') #seq 
        e_x = T.exp(a_2 - a_2.max(axis = -1, keepdims = True))
        #print e_x.shape.eval({c_i:np.random.rand(512,).astype('float32'), seq_i:np.random.rand(512,5).astype('float32')})
        #e_x[i] = 0
        e_x2 = T.set_subtensor(e_x[:,i], 0)
        e_x2 = T.flatten(e_x2)
        e_x = T.flatten(e_x)
        return e_x / e_x.sum(axis = -1, keepdims = True), e_x2 / e_x2.sum(axis = -1, keepdims = True)
    
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
        g, g_updates = theano.scan(fn=self.new_attention_step,
            sequences=self.inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(self.inp_c[0][0])) 
        
        if (self.normalize_attention):
            g = nn_utils.softmax(g)
        
        e, e_updates = theano.scan(fn=self.new_episode_step,
            sequences=[self.inp_c, g],
            outputs_info=T.zeros_like(self.inp_c[0]))
        
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
                    'word2vec': self.word2vec,
                    'baseline': self.baseline_time,
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
            self.baseline_time = dict['baseline']

        self.word2vec = dict['word2vec']
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

        max_q_len = 1 # just be 1.
        max_ans_len = 0
        for sid in stories:
            t_ans_len = 0
            for slid in split_dict_story[sid]:
                t_ans_len += len(slid[-1][-1])
            max_ans_len = max(max_ans_len, t_ans_len)
        
        max_ans_len += 1 # this is for the start token.
        # in our case, it is pretty similar to the word-level dmn,

        questions = []
        # batch x story_len x fea
        inputs = []
        answers = []
        answers_inp = []
        answers_mask = []
        answers_idx = []
        max_key_len = 12

        with split_lmdb_env_fc.begin() as txn_fc:
            with split_lmdb_env_conv.begin() as txn_conv:

                for sid in stories:
                    anno = split_dict_story[sid]
                    question = []
                    answer = []
                    answer_mask = []
                    answer_inp = []
                    answer_idx = []
                    inp = []
                    
                    for slid in split_dict_story[sid]:
                        input_anno = slid
                        img_id = input_anno[1][0]
                        while len(img_id) < max_key_len:
                            img_id = '0' + img_id

                        fc_raw = txn_fc.get(img_id.encode('ascii'))
                        fc_fea = caffe.proto.caffe_pb2.Datum()
                        fc_fea.ParseFromString(fc_raw)
                        question.append( np.fromstring(fc_fea.data, dtype = np.float32))

                        conv_raw = txn_conv.get(img_id.encode('ascii'))
                        conv_datum = caffe.proto.caffe_pb2.Datum()
                        conv_datum.ParseFromString(conv_raw)
                        conv_fea = np.fromstring(conv_datum.data, dtype = np.float32)
                        x = conv_fea.reshape(conv_datum.channels, conv_datum.height, conv_datum.width) # 512 x 14 x 14
                        x = x.reshape(conv_datum.channels, conv_datum.height * conv_datum.width)
                        x = x.swapaxes(0,1)
                        inp.append(x)
                        a = []
                        a.append(self.vocab_size) # start token.
                        a.extend(input_anno[1][2]) # this is the index for the captions.

                        a_inp = np.zeros((max_ans_len, self.word_vector_size), dtype = floatX)
                        a_mask = []
                        a_inp[0, :] = utils.process_word2( word = "#START#",
                                                                  word2vec = self.word2vec,
                                                                  vocab = self.vocab,
                                                                  word_vector_size = self.word_vector_size,
                                                                  to_return = 'word2vec' )

                        for ans_idx, w_idx in enumerate(a[1:]):
                            a_inp[ans_idx + 1, :] = utils.process_word2( word = self.ivocab[w_idx],
                                                                        word2vec = self.word2vec,
                                                                        vocab = self.vocab,
                                                                        word_vector_size = self.word_vector_size,
                                                                        to_return = 'word2vec' )


                        a_mask = [ 1 for i in range(len(a) -1) ]
                        answer_idx.append(len(a_mask))
                        while len(a) < max_ans_len: # this does not matter.
                            a.append( 0 )
                            a_mask.append(0)

                        a = a[1:]
                        answer.append(np.array(a).astype(np.int32))
                        answer_mask.append(np.array(a_mask).astype(np.int32))
                        answer_inp.append(a_inp)
                    question = np.stack(question, axis = 0)
                    questions.append(question)
                    inp = np.stack(inp, axis = 0) # #story_len x patches x fea
                    inputs.append(inp)
                    
                    answer = np.stack(answer, axis = 0) # story_len x max_answer_len
                    answers.append(answer)
                    answer_mask = np.stack(answer_mask, axis =0) # story_len x max_answer_len -1
                    answers_mask.append(answer_mask)
                    answer_inp = np.stack(answer_inp, axis = 0) # story_len x max_answer_len
                    answers_inp.append(answer_inp)
                    answers_idx.append(np.asarray(answer_idx))

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
        #questions = np.reshape(questions, (questions.shape[0] * questions.shape[1], questions.shape[2]))
        answers = np.reshape(answers, (answers.shape[0] * answers.shape[1], answers.shape[2]))
        answers = np.reshape(answers, (answers.size,))
        answers_inp = np.reshape(answers_inp, (answers_inp.shape[0] * answers_inp.shape[1], answers_inp.shape[2], answers_inp.shape[3]))
        answers_mask = np.reshape(answers_mask, (answers_mask.shape[0] * answers_mask.shape[1], answers_mask.shape[2]))
        answers_idx = np.stack(answers_idx, axis = 0).astype('int32')

        #print 'input.shape', inputs.shape
        #print 'quesionts.shape', questions.shape
        #print 'answers.shape', answers.shape
        #print 'answers_inp.shape', answers_inp.shape
        #print 'answers_mask.shape', answers_mask.shape
        #print 'answers_idx.shape', answers_idx.shape
        
        return inputs, questions, answers, answers_inp, answers_mask,answers_idx, stories
 
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

    def _process_input_sind_lmdb(self, data_dir, split = 'train'):

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
            story = story[::-1]
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
    
    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            cnt = 0
            cnt = len(self.train_dict_story)
            return cnt / self.batch_size
        elif (mode == 'test'):
            cnt = len(self.test_dict_story)
            return cnt /self.batch_size
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
        
        inp, q, ans, ans_inp, ans_mask, ans_idx, img_ids = self._process_batch_sind(batch_index, mode)
        
        ret = theano_fn(inp, q, ans, ans_mask, ans_inp, ans_idx)
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"prediction": ret[0],
                "answers": ans,
                "current_loss": ret[1],
                "skipped": 0,
                "log": "pn: %.3f" % param_norm,
                }

    def step_beam(self, batch_index, beam_size = 4):
        '''
            This function is mainly for the testing stage.
            Use the beam search to generate the target captions from each image.
        '''

        theano_fn = self.beam_pred_fn
        #q, ans, ans_inp, ans_mask, img_ids = self._process_batch_sind(batch_index, 'test')
        inp, q, ans, ans_inp, ans_mask, ans_idx, img_ids = self._process_batch_sind(batch_index, 'test')
        
        batch_size = self.batch_size
        q_shape = q.shape[0]
        if q_shape < batch_size:
            q_ = np.zeros((self.batch_size, self.story_len, q.shape[2]), dtype = 'float32')
            inp_ = np.zeros((self.batch_size, self.story_len, self.patches, self.cnn_dim_fc), dtype = 'float32')
            ans_idx_ = np.zeros((self.batch_size, ans_idx.shape[1]), dtype = 'float32')
            q_[0:q_shape,:,:] = q
            inp_[0:q_shape,:,:,:] = inp
            ans_idx_[0:q_shape,:] = ans_idx

            q_[q_shape:,:,:] = q[0,:,:]
            inp_[q_shape:,:,:,:] = inp[0,:,:,:]
            ans_idx_[q_shape:,:] = ans_idx[0,:]
            q = q_
            inp = inp_
            ans_idx = ans_idx_

        init_h = None
        captions = []
        batch_of_beams = [ [ (0.0, [])]  for i in range(batch_size) ]
        beam_r = [ [] for i in range(batch_size) ]
        for story_idx in range(self.story_len):
            if story_idx == 0:
                init_h = np.zeros((self.dim, self.batch_size), dtype = 'float32')
            
            for bbeam in batch_of_beams:
                for beam in bbeam:
                    beam[1].append(self.vocab_size)
            #batch_of_beams = [ (0.0, [self.vocab_size])  for i in range(batch_size) ]
            nsteps = 0
            cur_beam_r = None
            if story_idx > 0:
                cur_beam_r = beam_r
                beam_r = [ [] for i in range(batch_size) ]

            while True:
                logging.info('nsteps = %d', nsteps)
                beam_c = [[] for i in range(batch_size) ]
                beam_r_c = [ [] for i in range(batch_size) ]
                idx_prevs = [ [] for i in range(batch_size)]
                idx_of_idx = [ [] for i in range(batch_size)]
                idx_of_idx_len = []

                max_b = -1
                cnt_ins = 0
                for i in range(batch_size):
                    beams = batch_of_beams[i]
                    for k, b in enumerate(beams):
                        idx_prev = b[-1]
                        if idx_prev[-1] == self.vocab['.']:
                            # This is the end.
                            beam_c[i].append(b)
                            beam_r_c[i].append(beam_r[i][k])
                            continue

                        idx_prevs[i].append( idx_prev )
                        idx_of_idx[i].append(k)
                        idx_of_idx_len.append( len(idx_prev))
                        cnt_ins += 1

                        if len(idx_prev) > max_b:
                            max_b = len(idx_prev)

                if cnt_ins == 0:
                    break
                
                x_i = np.zeros((cnt_ins, max_b, self.word_vector_size), dtype = 'float32')
                v_i = np.zeros((cnt_ins, self.story_len, inp.shape[2], self.cnn_dim), dtype = 'float32')
                q_i = np.zeros((cnt_ins, self.story_len, self.cnn_dim_fc), dtype='float32')
                beam_r_i = np.zeros((cnt_ins, self.dim), dtype = 'float32')

                idx_base = 0
                for j,idx_prev_j in enumerate(idx_prevs):
                    for m, idx_prev in enumerate(idx_prev_j):
                        if story_idx > 0:
                            #pdb.set_trace()
                            beam_r_i[idx_base+m,:] = cur_beam_r[j][m]
                        for k in range(len(idx_prev)):
                            if idx_prev[k] ==self.vocab_size:
                                x_i[m + idx_base,k,:] = utils.process_word2( word = "#START#",
                                                                  word2vec = self.word2vec,
                                                                  vocab = self.vocab,
                                                                  word_vector_size = self.word_vector_size,
                                                                  to_return = 'word2vec' )
                            else:
                                x_i[m + idx_base,k,:] = utils.process_word2( word = self.ivocab[idx_prev[k]],
                                                                        word2vec = self.word2vec,
                                                                        vocab = self.vocab,
                                                                        word_vector_size = self.word_vector_size,
                                                                        to_return = 'word2vec' )

                    q_i[idx_base:idx_base + len(idx_prev_j),:,:] = q[j,:,:]
                    v_i[idx_base:idx_base + len(idx_prev_j),:,:,:] = inp[j,:,:,:]
                    idx_base += len(idx_prev_j)

                # This is really pain full.
                # Since the batch_size is fixed when creating the module. Thus,
                # we need to make them equal to the batch_size.
                pred = np.zeros((cnt_ins, max_b, self.vocab_size + 1), dtype = 'float32')
                _beam_r = np.zeros((cnt_ins, max_b, self.dim), dtype = 'float32')
                for i in range(0, v_i.shape[0], batch_size):
                    start_idx = i
                    end_idx = i + batch_size
                    if end_idx > v_i.shape[0]:
                        end_idx = v_i.shape[0]
                        start_idx = max(end_idx - batch_size,0)

                    if end_idx - start_idx < batch_size:
                        t_q_i = np.zeros((batch_size, self.cnn_dim), dtype = 'float32')
                        t_x_i = np.zeros((batch_size, max_b, self.vocab_size + 1), dtype = 'float32')
                        t_v_i = np.zeros((batch_size, inp.shape[1], self.cnn_dim), dtype = 'float32')

                        t_q_i[0:(end_idx - start_idx),:] = q_i[start_idx:end_idx,:]
                        t_x_i[0:(end_idx - start_idx),:,:] = x_i[start_idx:end_idx,:,:]
                        t_v_i[0:(end_idx - start_idx),:,:] = v_i[start_idx:end_idx,:,:]
                        if story_idx > 0:
                            init_h = beam_r_i[start_idx:end_idx,:]
                            init_h = np.swapaxes(init_h, 0, 1)
                        t = theano_fn(t_v_i, t_q_i, t_x_i, init_h, story_idx)
                        pred[start_idx:end_idx,:,:] = t[0][0:(end_idx-start_idx),:,:]
                        _beam_r[start_idx:end_idx,:,:] = t[1][0:(end_idx - start_idx),:,:]
                    else:
                        #pdb.set_trace()
                        x_ii = x_i[start_idx:end_idx,:,:]
                        x_ii = x_ii[:,np.newaxis,:,:]
                        x_ii = np.repeat(x_ii, self.story_len, axis = 1)
                        x_ii = np.reshape(x_ii, (x_ii.shape[0] * x_ii.shape[1], x_ii.shape[2], x_ii.shape[3]))
                        #pdb.set_trace()
                        if story_idx > 0:
                            init_h = beam_r_i[start_idx:end_idx,:]
                            init_h = np.swapaxes(init_h, 0, 1)
                        t = theano_fn(v_i[start_idx:end_idx,:,:], q_i[start_idx:end_idx,:], x_ii, init_h, story_idx)

                        pred[start_idx:end_idx,:,:] = t[0]
                        _beam_r[start_idx:end_idx,:,:] = t[1]

                p = np.zeros((pred.shape[0], pred.shape[2]))
                _beam_r_p = np.zeros((_beam_r.shape[0], _beam_r.shape[2]))
                for i in range(pred.shape[0]):
                    p[i,:] = pred[i,idx_of_idx_len[i]-1,:]
                    _beam_r_p[i,:] = _beam_r[i, idx_of_idx_len[i]-1,:]

                l = np.log( 1e-20 + p)
                top_indices = np.argsort( -l, axis=-1)
                idx_base = 0
                for batch_i, idx_i in enumerate(idx_of_idx):
                    for j,idx in enumerate(idx_i):
                        row_idx = idx_base + j
                        for m in range(beam_size):
                            wordix = top_indices[row_idx][m]
                            beam_c[batch_i].append((batch_of_beams[batch_i][idx][0] + l[row_idx][wordix], batch_of_beams[batch_i][idx][1] + [wordix]))
                            beam_r_c[batch_i].append(_beam_r_p[row_idx,:])
                    idx_base += len(idx_i)

                for i in range(len(beam_c)):
                    loss = []
                    for li in beam_c[i]:
                        loss.append(-1 * li[0])
                    
                    order = np.argsort(loss)

                    batch_of_beams[i] = []
                    beam_r[i] = []
                    for j in range(beam_size):
                        batch_of_beams[i].append(beam_c[i][order[j]])
                        beam_r[i].append(beam_r_c[i][order[j]])
                    
                    #batch_of_beams[i] = beam_c[i][:beam_size]
                nsteps += 1
                if nsteps >= 10:
                    break
            
        for beams in batch_of_beams:
            pred = [(b[0], b[1]) for b in beams ]
            captions.append(pred)

        return {'captions':captions,
                'img_ids': img_ids}
