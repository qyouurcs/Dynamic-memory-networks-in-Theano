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

import sys
reload(sys)  
sys.setdefaultencoding('utf8')

class DMN_batch:
    
    def __init__(self, data_dir, word2vec, word_vector_size, dim, cnn_dim, story_len,
                mode, answer_module, memory_hops, batch_size, l2,
                normalize_attention, batch_norm, dropout, **kwargs):
        
        print "==> not used params in DMN class:", kwargs.keys()

        self.data_dir = data_dir
        
        self.word2vec = word2vec
        self.word_vector_size = word_vector_size
        self.dim = dim
        self.cnn_dim = cnn_dim
        self.story_len = story_len
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
        self.train_dict_story, self.train_lmdb_env_fc = self._process_input_sind_lmdb(self.data_dir, 'train')
        self.val_dict_story, self.val_lmdb_env_fc = self._process_input_sind_lmdb(self.data_dir, 'val')
        self.test_dict_story, self.test_lmdb_env_fc = self._process_input_sind_lmdb(self.data_dir, 'test')

        self.train_story = self.train_dict_story.keys()
        self.val_story = self.val_dict_story.keys()
        self.test_story = self.test_dict_story.keys()
        self.vocab_size = len(self.vocab)
        
        self.q_var = T.tensor3('q_var') # Now, it's a batch * story_len * image_sieze.
        self.answer_var = T.imatrix('answer_var') # answer of example in minibatch
        self.answer_mask = T.matrix('answer_mask')
        self.answer_inp_var = T.tensor3('answer_inp_var') # answer of example in minibatch
        
        print "==> building input module"
        # It's very simple now, the input module just need to map from cnn_dim to dim.
        logging.info('self.cnn_dim = %d', self.cnn_dim)
        self.W_inp_emb_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim))
        self.b_inp_emb_in = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        q_seq = self.q_var.dimshuffle(0,'x', 1, 2)
        q_seq_rpt = T.repeat(q_seq, self.story_len, 1)
        q_seq_rhp = T.reshape(q_seq_rpt, (q_seq_rpt.shape[0] * q_seq_rpt.shape[1], q_seq_rpt.shape[2], q_seq_rpt.shape[3]))

        inp_var_shuffled = q_seq_rhp.dimshuffle(1,2,0) #seq x cnn x batch
        def _dot(x, W, b):
            return  T.dot(W, x) + b.dimshuffle(0, 'x')

        inp_c_hist,_ = theano.scan(fn = _dot, sequences=inp_var_shuffled, non_sequences = [self.W_inp_emb_in, self.b_inp_emb_in])
        #inp_c_hist,_ = theano.scan(fn = _dot, sequences=self.input_var, non_sequences = [self.W_inp_emb_in, self.b_inp_emb_in])

        self.inp_c = inp_c_hist # seq x emb x batch

        print "==> building question module"
        # Now, share the parameter with the input module.
        q_var_shuffled = self.q_var.dimshuffle(1,2,0) # now: story_len * image_size * batch_size

        # This is the RNN used to produce the Global Glimpse
        self.W_inpf_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim))
        self.W_inpf_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inpf_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inpf_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim))
        self.W_inpf_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inpf_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inpf_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim))
        self.W_inpf_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inpf_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        inp_dummy = theano.shared(np.zeros((self.dim, self.batch_size), dtype = floatX))

        q_glb,_ = theano.scan(fn = self.input_gru_step_forward, 
                                    sequences = q_var_shuffled,
                                    outputs_info = [T.zeros_like(inp_dummy)])
        q_glb_shuffled = q_glb.dimshuffle(2,0,1) # batch_size * seq_len * dim
        q_glb_last = q_glb_shuffled[:,-1,:] # batch_size * dim

        # Now, we also need to build the individual model.
        #q_var_shuffled = self.q_var.dimshuffle(1,0)
        q_single = T.reshape(self.q_var, (self.q_var.shape[0] * self.q_var.shape[1], self.q_var.shape[2]))
        q_single_shuffled = q_single.dimshuffle(1,0) #cnn_dim x batch_size * 5 
        
        # batch_size * 5 x dim
        q_hist = T.dot(self.W_inp_emb_in, q_single_shuffled) + self.b_inp_emb_in.dimshuffle(0,'x')
        q_hist_shuffled = q_hist.dimshuffle(1,0) # batch_size * 5 x dim

        if self.batch_norm:
            logging.info("Using batch normalization.")
        q_net = layers.InputLayer(shape=(self.batch_size*self.story_len, self.dim), input_var=q_hist_shuffled)
        if self.batch_norm:
            q_net = layers.BatchNormLayer(incoming=q_net)
        if self.dropout > 0 and self.mode == 'train':
            q_net = layers.DropoutLayer(q_net, p=self.dropout)
        #last_mem = layers.get_output(q_net).dimshuffle((1, 0))
        self.q_q = layers.get_output(q_net).dimshuffle(1,0)

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
        
        self.W_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
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

        print "==> building answer module"

        answer_inp_var_shuffled = self.answer_inp_var.dimshuffle(1,2,0)
        # Sounds good. Now, we need to map last_mem to a new space. 
        self.W_mem_emb = nn_utils.normal_param(std = 0.1, shape = (self.dim, self.dim * 3))
        self.W_inp_emb = nn_utils.normal_param(std = 0.1, shape = (self.dim, self.vocab_size + 1))

        def _dot2(x, W):
            return  T.dot(W, x)

        answer_inp_var_shuffled_emb,_ = theano.scan(fn = _dot2, sequences = answer_inp_var_shuffled,
                non_sequences = self.W_inp_emb ) # seq x dim x batch
        
        # dim x batch_size * 5
        q_glb_dim = q_glb_last.dimshuffle(0,'x', 1) # batch_size * 1 * dim
        q_glb_repmat = T.repeat(q_glb_dim, self.story_len, 1) # batch_size * len * dim
        q_glb_rhp = T.reshape(q_glb_repmat, (q_glb_repmat.shape[0] * q_glb_repmat.shape[1], q_glb_repmat.shape[2]))
        init_ans = T.concatenate([self.q_q, last_mem, q_glb_rhp.dimshuffle(1,0)], axis = 0)

        mem_ans = T.dot(self.W_mem_emb, init_ans) # dim x batchsize.
        mem_ans_dim = mem_ans.dimshuffle('x',0,1)
        answer_inp = T.concatenate([mem_ans_dim, answer_inp_var_shuffled_emb], axis = 0)
        
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

        results, _ = theano.scan(fn = self.answer_gru_step,
                sequences = answer_inp,
                outputs_info = [ dummy ])
        # Assume there is a start token 
        #print results.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
        #    self.q_var: np.random.rand(10, 4096).astype('float32'), 
        #    self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')}, on_unused_input='ignore')
        #results = results[1:-1,:,:] # get rid of the last token as well as the first one (image)
        #print results.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
        #    self.q_var: np.random.rand(10, 4096).astype('float32'), 
        #    self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')}, on_unused_input='ignore')
            
        # Now, we need to transform it to the probabilities.

        prob,_ = theano.scan(fn = lambda x, w: T.dot(w, x), sequences = results, non_sequences = self.W_a )
        preds = prob[1:,:,:]
        prob = prob[1:-1,:,:]

        prob_shuffled = prob.dimshuffle(2,0,1) # b * len * vocab
        preds_shuffled = preds.dimshuffle(2,0,1)


        logging.info("prob shape.")
        #print prob.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
        #    self.q_var: np.random.rand(10, 4096).astype('float32'), 
        #    self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')})

        n = prob_shuffled.shape[0] * prob_shuffled.shape[1]
        n_preds = preds_shuffled.shape[0] * preds_shuffled.shape[1]

        prob_rhp = T.reshape(prob_shuffled, (n, prob_shuffled.shape[2]))
        preds_rhp = T.reshape(preds_shuffled, (n_preds, preds_shuffled.shape[2]))

        prob_sm = nn_utils.softmax_(prob_rhp)
        preds_sm = nn_utils.softmax_(preds_rhp)
        self.prediction = prob_sm # this one is for the training.

        # This one is for the beamsearch.
        self.pred = T.reshape(preds_sm, (preds_shuffled.shape[0], preds_shuffled.shape[1], preds_shuffled.shape[2]))

        mask =  T.reshape(self.answer_mask, (n,))
        lbl = T.reshape(self.answer_var, (n,))

        self.params = [self.W_inp_emb_in, self.b_inp_emb_in, 
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid, #self.W_b
                  self.W_1, self.W_2, self.b_1, self.b_2, self.W_a]
        
        self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid,
                              self.W_mem_emb, self.W_inp_emb]
                              
                              
        print "==> building loss layer and computing updates"
        loss_vec = T.nnet.categorical_crossentropy(prob_sm, lbl)
        self.loss_ce = (mask * loss_vec ).sum() / mask.sum() 

        #self.loss_ce = T.nnet.categorical_crossentropy(results_rhp, lbl)
            
        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = self.loss_ce + self.loss_l2
            
        updates = lasagne.updates.adadelta(self.loss, self.params)
        #updates = lasagne.updates.momentum(self.loss, self.params, learning_rate=0.001)
        
        if self.mode == 'train':
            print "==> compiling train_fn"
            self.train_fn = theano.function(inputs=[self.q_var, self.answer_var, self.answer_mask, self.answer_inp_var], 
                                            outputs=[self.prediction, self.loss],
                                            updates=updates)
        
        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=[self.q_var, self.answer_var, self.answer_mask, self.answer_inp_var],
                                       outputs=[self.prediction, self.loss])
        
        print "==> compiling pred_fn"
        self.pred_fn= theano.function(inputs=[self.q_var, self.answer_inp_var],
                                       outputs=[self.pred])
    
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
    
    def input_gru_step_forward(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inpf_res_in, self.W_inpf_res_hid, self.b_inpf_res, 
                                     self.W_inpf_upd_in, self.W_inpf_upd_hid, self.b_inpf_upd,
                                     self.W_inpf_hid_in, self.W_inpf_hid_hid, self.b_inpf_hid)

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
            split_story = self.train_story
            split_dict_story = self.train_dict_story
        elif split == 'test':
            split_lmdb_env_fc = self.test_lmdb_env_fc
            split_story = self.test_story
            split_dict_story = self.test_dict_story
        else:
            split_lmdb_env_fc = self.val_lmdb_env_fc
            split_story = self.val_story
            split_dict_story = self.val_dict_story

        # make sure it's small than the number of stories.
        start_index = start_index % len(split_story)
        # make sure there is enough for a batch.
        start_index = min(start_index, len(split_story) - self.batch_size)
        # Now, we select the stories.
        stories = split_story[start_index:start_index+self.batch_size]
        stories_rtn = [ s for s in stories for i in range(self.story_len)]
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
        answers = []
        answers_inp = []
        answers_mask = []
        max_key_len = 12

        with split_lmdb_env_fc.begin() as txn_fc:
            for sid in stories:
                anno = split_dict_story[sid]
                question = []
                answer = []
                
                answer.append(self.vocab_size)
                for slid in split_dict_story[sid]:
                    input_anno = slid
                    img_id = input_anno[1][0]
                    while len(img_id) < max_key_len:
                        img_id = '0' + img_id

                    fc_raw = txn_fc.get(img_id.encode('ascii'))
                    fc_fea = caffe.proto.caffe_pb2.Datum()
                    fc_fea.ParseFromString(fc_raw)
                    question.append( np.fromstring(fc_fea.data, dtype = np.float32))
                    answer.extend(input_anno[1][2]) # this is the index for the captions.
                
                answer_inp = np.zeros((max_ans_len, self.vocab_size + 1), dtype = floatX)
                answer_mask = []

                for ans_idx, w_idx in enumerate(answer):
                    answer_inp[ans_idx, w_idx] = 1

                answer_mask = [ 1 for i in range(len(answer) - 1)]
                

                while len(answer) < max_ans_len:
                    answer.append(-1)
                    answer_mask.append(0)
                
                answer = answer[1:]
                answers.append( np.array(answer).astype(np.int32))
                questions.append(np.stack(question, axis = 0))
                answers_inp.append(answer_inp)
                answers_mask.append(np.array(answer_mask).astype(np.int32))
                
        questions = np.stack(questions, axis = 0).astype(floatX)
        answers = np.array(answers).astype(np.int32)
        answers_rhp = np.reshape(answers, (answers.shape[0], 1, answers.shape[1]))
        answers = np.tile(answers_rhp, (1, questions.shape[1], 1))
        answers = np.reshape(answers, (answers.shape[0] * answers.shape[1], answers.shape[2]))
        answers_mask = np.array(answers_mask).astype(floatX)
        answers_mask_rhp = np.reshape(answers_mask, (answers_mask.shape[0], 1, answers_mask.shape[1]))
        answers_mask = np.tile(answers_mask_rhp, (1, questions.shape[1], 1))
        answers_mask = np.reshape(answers_mask, (answers_mask.shape[0] * answers_mask.shape[1], answers_mask.shape[2]))
        answers_inp = np.stack(answers_inp, axis = 0)
        answers_inp_rhp = np.reshape(answers_inp, (answers_inp.shape[0], 1, answers_inp.shape[1], answers_inp.shape[2]))
        answers_inp = np.tile(answers_inp_rhp, (1, questions.shape[1], 1, 1))
        answers_inp = np.reshape(answers_inp, (answers_inp.shape[0] * answers_inp.shape[1], answers_inp.shape[2], answers_inp.shape[3]))
        
        return questions, answers, answers_inp, answers_mask, stories_rtn
 
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

        lmdb_env_fc = lmdb.open(lmdb_dir_fc, readonly = True)

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

        return dict_story, lmdb_env_fc

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
            return cnt /self.batch_size + 1
        elif (mode == 'val'):
            cnt = len(self.val_dict_story)
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
        if mode == "test" or mode == 'val':    
            theano_fn = self.test_fn 
        
        q, ans, ans_inp, ans_mask, img_ids = self._process_batch_sind(batch_index, mode)
        
        ret = theano_fn(q, ans, ans_mask, ans_inp)
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"prediction": ret[0],
                "skipped": 0,
                "log": "pn: %.3f" % param_norm,
                }

    def step_beam(self, batch_index, beam_size = 5):
        '''
            This function is mainly for the testing stage.
            Use the beam search to generate the target captions from each image.
        '''

        theano_fn = self.pred_fn
        q, ans, ans_inp, ans_mask, img_ids = self._process_batch_sind(batch_index, 'test')
        
        batch_size = self.batch_size
        q_shape = q.shape[0]
        if q_shape < batch_size:
            q_ = np.zeros((self.batch_size, self.story_len, q.shape[2]), dtype = 'float32')
            q_[0:q_shape,:,:] = q
            q_[q_shape:,:,:] = q[0,:,:]
            q = q_

        captions = []
        batch_of_beams = [ [ [ (0.0, [self.vocab_size])] for j in range(self.story_len)] for i in range(batch_size) ]

        nsteps = 0

        while True:
            #logging.info('nsteps = %d', nsteps)
            beam_c = [[[] for j in range(self.story_len)] for i in range(batch_size) ]
            idx_prevs = [ [ [] for j in range(self.story_len)] for i in range(batch_size)]
            idx_of_idx = [[ [] for j in range(self.story_len)] for i in range(batch_size)]
            idx_of_idx_len = [ [[] for j in range(self.story_len)] for i in range( batch_size)]

            max_b = -1
            max_story = 0
            max_num_beam = 0

            is_done = True
            #pdb.set_trace()
            for i in range(batch_size):
                for j in range(self.story_len):
                    beams = batch_of_beams[i][j]
                    for k, b in enumerate(beams):
                        idx_prev = b[-1]
                        if idx_prev[-1] == self.vocab['.']:
                            # This is the end.
                            beam_c[i][j].append(b)
                            continue

                        idx_prevs[i][j].append( idx_prev )
                        idx_of_idx[i][j].append(k)
                        idx_of_idx_len[i][j].append( len(idx_prev))
                        #cnt_ins += 1
                        is_done = False

                        if len(idx_prev) > max_b:
                            max_b = len(idx_prev)
                    max_num_beam = max( len(idx_prevs[i][j]), max_num_beam)

            if is_done:
                break
            # Now, we need to manually make them aligned. 
            # In all cases, we should fix the q_i, which is all the stories of the current batch.

            x_i = np.zeros((batch_size, max_b, self.vocab_size + 1), dtype = 'float32')
            #q_i = np.zeros((batch_size, self.story_len, self.cnn_dim), dtype='float32')
            num_beam_cnt = [ [ 0 for j in range(self.story_len) ] for i in range(batch_size)]
            for i in range(batch_size):
                for j in range(self.story_len):
                    num_beam_cnt[i][j] = len(idx_prevs[i][j])
                    while len(idx_prevs[i][j]) < max_num_beam:
                        idx_prevs[i][j].append(  [self.vocab_size])
                        idx_of_idx[i][j].append( len( idx_of_idx[i][j]))
                        idx_of_idx_len[i][j].append(1)

            pred = np.zeros((max_num_beam, self.batch_size * self.story_len, self.vocab_size + 1), dtype = 'float32')
            for i in range(max_num_beam):
                x_i = np.zeros((batch_size, self.story_len, max_b, self.vocab_size + 1), dtype = 'float32')
                for j in range(batch_size):
                    for k in range(self.story_len):
                        idx_prev = idx_prevs[j][k][i]
                        for m in range(len(idx_prev)):
                            x_i[j,k, m,idx_prev[m]] = 1.0
                x_i = np.reshape(x_i, (x_i.shape[0] * x_i.shape[1], x_i.shape[2], x_i.shape[3]))
                t = theano_fn(q, x_i)[0]
                for j in range(t.shape[0]):
                    pred[i,j,:] = t[j, idx_of_idx_len[j / self.story_len][j % self.story_len][i] - 1,:]

            #pdb.set_trace()
            for i in range(max_num_beam):
                l = np.log( 1e-20 + pred[i,:])
                top_indices = np.argsort( -l, axis=-1)
                l =  np.reshape(l, (batch_size, self.story_len, self.vocab_size + 1))
                top_indices = np.reshape(top_indices, (batch_size, self.story_len, self.vocab_size+1))

                for j in range(batch_size):
                    for k in range(self.story_len):
                        if num_beam_cnt[j][k] < i + 1:
                            continue
                        for m in range(beam_size):
                            wordix = top_indices[j][k][m]
                            beam_c[j][k].append((batch_of_beams[j][k][idx_of_idx[j][k][i]][0] + l[j][k][wordix], \
                                batch_of_beams[j][k][idx_of_idx[j][k][i]][1] + [wordix]))
            for j in range(batch_size):
                for k in range(self.story_len):
                    c = beam_c[j][k][0:beam_size * beam_size]
                    #print i,j,len(c)
                    c.sort(reverse = True)
                    batch_of_beams[j][k] = c[0:beam_size]

            nsteps += 1
            if nsteps >= 20:
                break
            
        for j in range(batch_size):
            for k in range( self.story_len):
                pred = [ (b[0], b[1]) for b in batch_of_beams[j][k] ]
                captions.append(pred)

        return {'captions':captions,
                'img_ids': img_ids}
