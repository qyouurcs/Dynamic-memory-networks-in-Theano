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
    
    def __init__(self, data_dir, word2vec, truncate_gradient, learning_rate, dim, cnn_dim, story_len,
                mode, answer_module, batch_size, l2,
                dropout, **kwargs):
        
        print "==> not used params in DMN class:", kwargs.keys()

        self.data_dir = data_dir
        self.learning_rate = learning_rate
        self.word_vector_size = 300

        self.truncate_gradient = truncate_gradient
        self.word2vec = word2vec
        self.dim = dim
        self.cnn_dim = cnn_dim
        self.story_len = story_len
        self.mode = mode
        self.answer_module = answer_module
        self.batch_size = batch_size
        self.l2 = l2
        self.dropout = dropout

        self.vocab, self.ivocab = self._load_vocab(self.data_dir)

        self.train_story = None
        self.test_story = None
        self.train_dict_story, self.train_lmdb_env_fc = self._process_input_sind(self.data_dir, 'train')
        self.test_dict_story, self.test_lmdb_env_fc = self._process_input_sind(self.data_dir, 'val')

        self.train_story = self.train_dict_story.keys()
        self.test_story = self.test_dict_story.keys()
        self.vocab_size = len(self.vocab)

        self.q_var = T.tensor3('q_var') # Now, it's a batch * story_len * image_sieze.
        self.answer_var = T.imatrix('answer_var') # answer of example in minibatch
        self.answer_mask = T.matrix('answer_mask')
        self.answer_inp_var = T.tensor3('answer_inp_var') # answer of example in minibatch

        q_shuffled = self.q_var.dimshuffle(1,2,0) # now: story_len * image_size * batch_size
        
        print "==> building input module"

        # Now, we use a GRU to produce the input.
        # Forward GRU.
        self.W_inpf_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim))
        self.W_inpf_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inpf_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inpf_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim))
        self.W_inpf_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inpf_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inpf_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.cnn_dim))
        self.W_inpf_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inpf_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        # Now, we use the GRU to build the inputs.
        # Two-level of nested scan is unnecessary. It will become too complicated. Just use this one.
        inp_dummy = theano.shared(np.zeros((self.dim, self.batch_size), dtype = floatX))

        q_inp,_ = theano.scan(fn = self.input_gru_step_forward, 
                                    sequences = q_shuffled,
                                    outputs_info = [T.zeros_like(inp_dummy)])
        q_inp_shuffled = q_inp.dimshuffle(2,0,1)
        q_inp_last = q_inp_shuffled[:,-1,:].dimshuffle(1,0) #dim * batch_size
        

        # Now, share the parameter with the input module.
        self.W_inp_emb_q = nn_utils.normal_param(std = 0.1, shape=(self.dim, self.dim))
        self.b_inp_emb_q = nn_utils.normal_param(std = 0.1, shape=(self.dim,))

        inp_q = T.dot(self.W_inp_emb_q, q_inp_last) + self.b_inp_emb_q.dimshuffle(0,'x') # 512 x 50
        self.q_q = T.tanh(inp_q) # Since this is used to initialize the memory, we need to make it tanh.
        
        print "==> building answer module"

        answer_inp_var_shuffled = self.answer_inp_var.dimshuffle(1,2,0)
        # Sounds good. Now, we need to map last_mem to a new space. 
        self.W_inp_emb = nn_utils.normal_param(std = 0.1, shape = (self.dim, self.vocab_size + 1))

        def _dot2(x, W):
            return  T.dot(W, x)

        answer_inp_var_shuffled_emb,_ = theano.scan(fn = _dot2, sequences = answer_inp_var_shuffled,
                non_sequences = self.W_inp_emb ) # seq x dim x batch
        
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
                sequences = answer_inp_var_shuffled_emb,
                outputs_info = [ self.q_q ])
        # Assume there is a start token 
        #print results.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
        #    self.q_var: np.random.rand(10, 4096).astype('float32'), 
        #    self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')}, on_unused_input='ignore')
        #results = results[:-1,:,:] # get rid of the last token as well as the first one (image)
        #print results.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
        #    self.q_var: np.random.rand(10, 4096).astype('float32'), 
        #    self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')}, on_unused_input='ignore')
            
        # Now, we need to transform it to the probabilities.

        prob_f,_ = theano.scan(fn = lambda x, w: T.dot(w, x), sequences = results, non_sequences = self.W_a )
        prob = prob_f[:-1,:,:]

        prob_shuffled = prob.dimshuffle(2,0,1) # b * len * vocab
        prob_f_shuffled = prob_f.dimshuffle(2,0,1)


        logging.info("prob shape.")
        #print prob.shape.eval({self.input_var: np.random.rand(10,4,4096).astype('float32'),
        #    self.q_var: np.random.rand(10, 4096).astype('float32'), 
        #    self.answer_inp_var: np.random.rand(10, 18, 8001).astype('float32')})

        n = prob_shuffled.shape[0] * prob_shuffled.shape[1]
        prob_rhp = T.reshape(prob_shuffled, (n, prob_shuffled.shape[2]))
        prob_sm = nn_utils.softmax_(prob_rhp)
        self.pred = prob_sm

        n_f = prob_f_shuffled.shape[0] * prob_f_shuffled.shape[1]
        prob_f_rhp = T.reshape(prob_f_shuffled, (n_f, prob_f_shuffled.shape[2]))

        prob_f_sm = nn_utils.softmax_(prob_f_rhp)
        self.prediction = T.reshape(prob_f_sm, (prob_f_shuffled.shape[0], prob_f_shuffled.shape[1], prob_f_shuffled.shape[2]))

        mask =  T.reshape(self.answer_mask, (n,))
        lbl = T.reshape(self.answer_var, (n,))

        self.params = [
                  self.W_inpf_res_in, self.W_inpf_res_hid,self.b_inpf_res,
                  self.W_inpf_upd_in, self.W_inpf_upd_hid, self.b_inpf_upd,
                  self.W_inpf_hid_in, self.W_inpf_hid_hid, self.b_inpf_hid,
                  self.W_inp_emb_q, self.b_inp_emb_q,
                  self.W_a,
                  self.W_inp_emb,
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
            self.train_fn = theano.function(inputs=[self.q_var, self.answer_var, self.answer_mask, self.answer_inp_var], 
                                            outputs=[self.pred, self.loss],
                                            updates=updates)
        
        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=[self.q_var, self.answer_var, self.answer_mask, self.answer_inp_var],
                                       outputs=[self.pred, self.loss])
        
        print "==> compiling pred_fn"
        self.pred_fn = theano.function(inputs=[self.q_var, self.answer_inp_var],
                                       outputs=[self.prediction])
    
    
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
        else:
            split_lmdb_env_fc = self.test_lmdb_env_fc
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
        answers_mask = np.array(answers_mask).astype(floatX)
        answers_inp = np.stack(answers_inp, axis = 0)

        return questions, answers, answers_inp, answers_mask, stories
   
    
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
        
        q, ans, ans_inp, ans_mask, stories = self._process_batch_sind(batch_index, mode)
        
        ret = theano_fn(q, ans, ans_mask, ans_inp)
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"prediction": ret[0],
                "answers": ans,
                "current_loss": ret[1],
                "skipped": 0,
                "log": "pn: %.3f" % param_norm,
                }
        
    def step_beam(self, batch_index, beam_size = 10):
        '''
            This function is mainly for the testing stage.
            Use the beam search to generate the target captions from each image.
        '''

        theano_fn = self.pred_fn
        q, ans, ans_inp, ans_mask, img_ids = self._process_batch_sind(batch_index)
        
        batch_size = q.shape[0]

        captions = []
        batch_of_beams = [ [ (0.0, [self.vocab_size])] for i in range(batch_size) ]

        nsteps = 0

        while True:
            logging.info('nsteps = %d', nsteps)
            beam_c = [[] for i in range(batch_size) ]
            idx_prevs = [ [] for i in range(batch_size)]
            idx_of_idx = [[] for i in range(batch_size)]
            idx_of_idx_len = [ ]

            max_b = -1
            cnt_ins = 0
            for i in range(batch_size):
                beams = batch_of_beams[i]
                for k, b in enumerate(beams):
                    idx_prev = b[-1]
                    #if idx_prev[-1] == self.vocab['.']:
                    #    # This is the end.
                    #    beam_c[i].append(b)
                    #    continue

                    idx_prevs[i].append( idx_prev )
                    idx_of_idx[i].append(k)
                    idx_of_idx_len.append( len(idx_prev))
                    cnt_ins += 1

                    if len(idx_prev) > max_b:
                        max_b = len(idx_prev)

            if cnt_ins == 0:
                break
            
            x_i = np.zeros((cnt_ins, max_b, self.vocab_size + 1), dtype = 'float32')
            q_i = np.zeros((cnt_ins, self.story_len, self.cnn_dim), dtype='float32')

            idx_base = 0
            for j,idx_prev_j in enumerate(idx_prevs):
                for m, idx_prev in enumerate(idx_prev_j):
                    for k in range(len(idx_prev)):
                        x_i[m + idx_base,k,idx_prev[k]] = 1.0
                q_i[idx_base:idx_base + len(idx_prev_j),:,:] = q[j,:,:]

                idx_base += len(idx_prev_j)
            # This is really pain full.
            # Since the batch_size is fixed when creating the module. Thus,
            # we need to make them equal to the batch_size.
            pred = np.zeros_like(x_i)
            for i in range(0, cnt_ins, batch_size):
                start_idx = i
                end_idx = i + batch_size
                if end_idx > cnt_ins:
                    end_idx = cnt_ins
                    start_idx = max(end_idx - batch_size,0)

                if end_idx - start_idx < batch_size:
                    t_q_i = np.zeros((batch_size, self.story_len, self.cnn_dim), dtype = 'float32')
                    t_x_i = np.zeros((batch_size, max_b, self.vocab_size + 1), dtype = 'float32')
                    t_q_i[0:(end_idx - start_idx),:,:] = q_i[start_idx:end_idx,:,:]
                    t_x_i[0:(end_idx - start_idx),:,:] = x_i[start_idx:end_idx,:,:]
                    t = theano_fn(t_q_i, t_x_i)
                    pred[start_idx:end_idx,:,:] = t[0][0:(end_idx-start_idx),:,:]
                else:
                    t = theano_fn(q_i[start_idx:end_idx,:,:], x_i[start_idx:end_idx,:,:])
                    pred[start_idx:end_idx,:,:] = t[0]

            p = np.zeros((pred.shape[0], pred.shape[2]))
            for i in range(pred.shape[0]):
                p[i,:] = pred[i,idx_of_idx_len[i]-1,:]

            l = np.log( 1e-20 + p)
            top_indices = np.argsort( -l, axis=-1)
            idx_base = 0
            for batch_i, idx_i in enumerate(idx_of_idx):
                for j,idx in enumerate(idx_i):
                    row_idx = idx_base + j
                    for m in range(beam_size):
                        wordix = top_indices[row_idx][m]
                        beam_c[batch_i].append((batch_of_beams[batch_i][idx][0] + l[row_idx][wordix], batch_of_beams[batch_i][idx][1] + [wordix]))
                idx_base += len(idx_i)
                
            for i in range(len(beam_c)):
                beam_c[i].sort(reverse = True) # descreasing order.
            for i, b in enumerate(beam_c):
                batch_of_beams[i] = beam_c[i][:beam_size]
            nsteps += 1
            if nsteps >= 60:
                break
        for beams in batch_of_beams:
            pred = [(b[0], b[1]) for b in beams ]
            captions.append(pred)

        return {'captions':captions,
                'img_ids': img_ids}
