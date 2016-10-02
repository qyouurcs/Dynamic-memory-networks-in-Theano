import random
import numpy as np

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
    
    def __init__(self, data_dir, word2vec, word_vector_size, dim,
                mode, answer_module, memory_hops, batch_size, l2,
                normalize_attention, batch_norm, dropout, **kwargs):
        
        print "==> not used params in DMN class:", kwargs.keys()

        self.data_dir = data_dir
        
        self.word2vec = word2vec
        self.word_vector_size = word_vector_size
        self.dim = dim
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
        self.train_dict_story, self.train_features, self.train_fns_dict, self.train_num_imgs = self._process_input_sind(self.data_dir, 'train')
        self.test_dict_story, self.test_features, self.test_fns_dict, self.test_num_imgs = self._process_input_sind(self.data_dir, 'val')

        self.train_story = self.train_dict_story.keys()
        self.test_story = self.test_dict_story.keys()
        self.vocab_size = len(self.vocab)
        
        self.input_var = T.tensor3('input_var') # (batch_size, seq_len, cnn_dim)
        self.q_var = T.matrix('q_var') # Now, it's a batch * image_sieze.
        self.answer_var = T.imatrix('answer_var') # answer of example in minibatch
        self.answer_mask = T.matrix('answer_mask')
        self.answer_inp_var = T.tensor3('answer_inp_var') # answer of example in minibatch
        
        print "==> building input module"
        # It's very simple now, the input module just need to map from cnn_dim to dim.
        logging.info('self.cnn_dim = %d', self.cnn_dim)
        self.W_inp_emb_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim))
        self.b_inp_emb_in = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        #inp_c_hist = T.dot(self.W_inp_emb_in, self.input_var) + self.b_inp_emb_in
        inp_var_shuffled = self.input_var.dimshuffle(1,2,0)
        print inp_var_shuffled.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32')})
        def _dot(x, W, b):
            return  T.dot(W, x) + b.dimshuffle(0, 'x')

        inp_c_hist,_ = theano.scan(fn = _dot, sequences=inp_var_shuffled, non_sequences = [self.W_inp_emb_in, self.b_inp_emb_in])
        #inp_c_hist,_ = theano.scan(fn = _dot, sequences=self.input_var, non_sequences = [self.W_inp_emb_in, self.b_inp_emb_in])

        #self.inp_c = inp_c_hist.dimshuffle(2,0,1) # b x len x fea
        self.inp_c = inp_c_hist

        print "==> building question module"
        # Now, share the parameter with the input module.
        q_var_shuffled = self.q_var.dimshuffle(1,0)
        q_hist = T.dot(self.W_inp_emb_in, q_var_shuffled) + self.b_inp_emb_in.dimshuffle(0,'x')

        self.q_q = q_hist.dimshuffle(0,1) # batch x dim
        
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
        
        net = layers.InputLayer(shape=(self.batch_size, self.dim), input_var=last_mem_raw)

        if self.batch_norm:
            net = layers.BatchNormLayer(incoming=net)
        if self.dropout > 0 and self.mode == 'train':
            net = layers.DropoutLayer(net, p=self.dropout)
        last_mem = layers.get_output(net).dimshuffle((1, 0))

        logging.info('last_mem size')
        print last_mem.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
            self.q_var: np.random.rand(10, 4096).astype('float32')})
       
        print "==> building answer module"

        answer_inp_var_shuffled = self.answer_inp_var.dimshuffle(1,2,0)
        # because we have the additional #start token. Thus, we need to add this +1 for all the parameters as well.
        dummy = theano.shared(np.zeros((self.vocab_size + 1, self.batch_size), dtype=floatX))

        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size + 1, self.dim))
        
        self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size +1))
        self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim * 2))
        self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size +1))
        self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim * 2))
        self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size +1))
        self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim * 2))
        self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        logging.info('answer_inp_var_shuffled size')

        print answer_inp_var_shuffled.shape.eval({self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')})
        
        #last_mem = printing.Print('prob_sm')(last_mem)
        results, _ = theano.scan(fn = self.answer_gru_step,
                sequences = answer_inp_var_shuffled,
                outputs_info = [ last_mem ])
        # Assume there is a start token 
        print results.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
            self.q_var: np.random.rand(10, 4096).astype('float32'), 
            self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')})
        results = results[0:-1,:,:] # get rid of the last token.
        print results.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
            self.q_var: np.random.rand(10, 4096).astype('float32'), 
            self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')})
            
        # Now, we need to transform it to the probabilities.

        prob,_ = theano.scan(fn = lambda x, w: T.dot(w, x), sequences = results, non_sequences = self.W_a )

        prob_shuffled = prob.dimshuffle(2,0,1) # b * len * vocab


        logging.info("prob shape.")
        print prob.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
            self.q_var: np.random.rand(10, 4096).astype('float32'), 
            self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')})

        n = prob_shuffled.shape[0] * prob_shuffled.shape[1]
        prob_rhp = T.reshape(prob_shuffled, (n, prob_shuffled.shape[2]))
        prob_sm = nn_utils.softmax(prob_rhp)
        self.prediction = prob_sm

        mask =  T.reshape(self.answer_mask, (n,))
        lbl = T.reshape(self.answer_var, (n,))

        self.params = [self.W_inp_emb_in, self.b_inp_emb_in, 
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid, #self.W_b
                  self.W_1, self.W_2, self.b_1, self.b_2, self.W_a]
        
        self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]
                              
                              
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
    
    def answer_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, T.concatenate([ x, self.q_q.T]), 
                                     self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
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
        for index in range(self.batch_size):
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
            split_story = self.train_story
            split_dict_story = self.train_dict_story
            features = self.train_features
            fns_dict = self.train_fns_dict
            
        else:
            split_story = self.test_story
            split_dict_story = self.test_dict_story
            features = self.test_features
            fns_dict = self.test_fns_dict

        # make sure it's small than the number of stories.
        start_index = start_index % len(split_story)
        # make sure there is enough for a batch.
        start_index = min(start_index, len(split_story) - self.batch_size)
        # Now, we select the stories.
        stories = split_story[start_index:start_index+self.batch_size]
        # For each story, we randomly select one as the question and the remaining as the facts.
        slids = []
        for sid in stories:
            slids.append( random.choice(range(len(split_dict_story[sid]))))

        max_inp_len = 0
        max_q_len = 1 # just be 1.
        max_ans_len = 0
        for slid, sid in zip(slids, stories):
            max_inp_len = max(max_inp_len, len(split_dict_story[sid])-1)
            #max_q_len = max(max_q_len, split_dict_story[sid][slid][1])
            max_ans_len = max(max_ans_len, len(split_dict_story[sid][slid][1][2]))
        
        max_ans_len += 1 # this is for the start token.
        # in our case, it is pretty similar to the word-level dmn,

        questions = []
        inputs = []
        answers = []
        answers_inp = []
        answers_mask = []

        for slid, sid in zip(slids,stories):
            anno = split_dict_story[sid]
            input_anno = anno[slid]
            img_id = input_anno[1][0]
            
            # Now, find the input visual features.
            questions.append( features[fns_dict[img_id]] )
            # Now, it is the inputs, we can use the other images other than current one.
            inp = []
            for s_slid_idx,s_slid in enumerate(anno):
                if s_slid_idx != slid:
                    img_id = s_slid[1][0]
                    inp.append(features[fns_dict[img_id]])
            inp = np.stack(inp, axis = 0)
            #print inp.shape
            inputs.append(inp)
            answer = []
            answer.append(self.vocab_size)
            answer.extend(input_anno[1][2]) # this is the index for the captions.
            answer_inp = np.zeros((max_ans_len, self.vocab_size + 1), dtype = floatX)
            answer_mask = []
            for ans_idx, w_idx in enumerate(answer):
                answer_inp[ans_idx, w_idx] = 1
            answer_mask = [ 1 for i in range(len(answer) -1) ]
            while len(answer) < max_ans_len: # this does not matter.
                answer.append( -1 )
                answer_mask.append(0)
            answer = answer[1:]
            answers.append(answer)
            answers_mask.append(answer_mask)
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

        #print 'inputs', inputs.shape
        #print 'questions', questions.shape
        #print 'answers',answers.shape
        #print 'answers_inp', answers_inp.shape
        #print 'answers_mask',answers_mask.shape
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

        split_dir = os.path.join(data_dir, split)
        fea_dir = os.path.join(split_dir, 'fea_vgg16_fc7')
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

        # Load all features into memory.
        features = None
        num_imgs = 0
        fns_dict = {}

        total_fea = 0
        total_fns = 0

        for root, dirs, fns in os.walk(fea_dir, followlinks = True):
            for fn in fns:
                full_fn = os.path.join(root, fn)
                hdf_f = h5py.File(full_fn,'r')
                fea = hdf_f['fea'][:]
                fns = hdf_f['fns'][:]
                total_fea += fea.shape[0]
                total_fns += fns.shape[0]
                assert( fea.shape[0] == fns.shape[0], "Should not happen, we have re-runed the feature extraction.")
                hdf_f.close()

        logging.info('total fea = %d, fns = %d', total_fea, total_fns)
        for root, dirs, fns in os.walk(fea_dir, followlinks=True):
            for fn in fns:
                full_fn = os.path.join(root, fn)
                hdf_f = h5py.File(full_fn,'r')
                fea = hdf_f['fea'][:]
                fns = hdf_f['fns'][:]
                hdf_f.close()

                if features is None:
                    shape = [total_fea]
                    self.cnn_dim = fea.size / fea.shape[0]
                    shape.extend(fea.shape[1:])
                    features = np.zeros(shape)
                    features[0:fea.shape[0],:] = fea
                else:
                    features[num_imgs:num_imgs+fea.shape[0],:] = fea
                for i in range(fns.shape[0]):
                    bfn = os.path.basename(fns[i])
                    key = os.path.splitext(bfn)[0]
                    key = key.split('_')[0]
                    fns_dict[key] = num_imgs
                    num_imgs += 1

        logging.info("Done loading features from %s", fea_dir)

        return dict_story, features, fns_dict, num_imgs
    def _load_vocab(self, data_dir):
        v_fn = os.path.join(data_dir, 'vocab.txt')
        vocab = {}
        ivocab = {}
        with open(v_fn, 'r') as fid:
            for aline in fid:
                parts = aline.strip().split()
                vocab[parts[1]] = int(parts[0])
                ivocab[int(parts[0])] = parts[1]
        return vocab, ivocab
    
    
    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            cnt = 0
            for story in self.train_dict_story:
                cnt += len(self.train_dict_story[story])
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
        
        inp, q, ans, ans_inp, ans_mask = self._process_batch_sind(batch_index)
        
        ret = theano_fn(inp, q, ans, ans_mask, ans_inp)
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"prediction": ret[0],
                "answers": ans,
                "current_loss": ret[1],
                "skipped": 0,
                "log": "pn: %.3f" % param_norm,
                }
        
