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

import sys
reload(sys)  
sys.setdefaultencoding('utf8')

class DMN_batch:
    
    def __init__(self, data_dir, word2vec, word_vector_size, dim, SEQUENCE_LENGTH,
                mode, answer_module, memory_hops, batch_size, l2,
                normalize_attention, batch_norm, dropout, **kwargs):
        
        print "==> not used params in DMN class:", kwargs.keys()

        self.data_dir = data_dir
        self.SEQUENCE_LENGTH  =  SEQUENCE_LENGTH 
        
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
        
        self.q_var = T.matrix('q_var') # Now, it's a batch * image_sieze.
        self.answer_var = T.imatrix('answer_var') # answer of example in minibatch
        self.answer_mask = T.matrix('answer_mask')
        self.answer_inp_var = T.imatrix('answer_inp_var') # answer of example in minibatch
        
        l_input_sentence = lasagne.layers.InputLayer((self.batch_size, self.SEQUENCE_LENGTH - 1))
        l_sentence_embedding = lasagne.layers.EmbeddingLayer(l_input_sentence,
                                                     input_size=len(self.vocab) + 1,
                                                     output_size=self.dim,
                                                    )
        l_input_cnn = lasagne.layers.InputLayer((self.batch_size, self.cnn_dim))
        l_cnn_embedding = lasagne.layers.DenseLayer(l_input_cnn, num_units=self.dim,
                                                    nonlinearity=lasagne.nonlinearities.identity)
        
        l_cnn_embedding = lasagne.layers.ReshapeLayer(l_cnn_embedding, ([0], 1, [1]))
        
        # the two are concatenated to form the RNN input with dim (self.batch_size, self.SEQUENCE_LENGTH, self.dim)
        l_rnn_input = lasagne.layers.ConcatLayer([l_cnn_embedding, l_sentence_embedding])
        
        l_dropout_input = lasagne.layers.DropoutLayer(l_rnn_input, p=0.5)
        l_lstm = lasagne.layers.LSTMLayer(l_dropout_input,
                                          num_units=self.dim,
                                          unroll_scan=True,
                                          grad_clipping=5.)
        l_dropout_output = lasagne.layers.DropoutLayer(l_lstm, p=0.5)
        
        # the RNN output is reshaped to combine the batch and time dimensions
        # dim (self.batch_size * self.SEQUENCE_LENGTH, self.dim)
        l_shp = lasagne.layers.ReshapeLayer(l_dropout_output, (-1, self.dim))
        
        # decoder is a fully connected layer with one output unit for each word in the vocabulary
        l_decoder = lasagne.layers.DenseLayer(l_shp, num_units=len(self.vocab) + 1, nonlinearity=lasagne.nonlinearities.softmax)
        
        # finally, the separation between batch and time dimension is restored
        l_out = lasagne.layers.ReshapeLayer(l_decoder, (self.batch_size, self.SEQUENCE_LENGTH, len(self.vocab) + 1))
        
        
        
        output = lasagne.layers.get_output(l_out, {
                        l_input_sentence: self.answer_inp_var,
                        l_input_cnn: self.q_var 
        })
        
        
        def calc_cross_ent(net_output, mask, targets):
            # Helper function to calculate the cross entropy error
            preds = T.reshape(net_output, (-1, len(self.vocab) + 1))
            targets = T.flatten(targets)
            cost = T.nnet.categorical_crossentropy(preds, targets)[T.flatten(mask).nonzero()]
            return cost
        
        self.loss_ce = T.mean(calc_cross_ent(output, self.answer_mask, self.answer_var))
 
        MAX_GRAD_NORM = 1e5
        
        self.params = lasagne.layers.get_all_params(l_out, trainable=True)

        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = self.loss_ce + self.loss_l2
       
        all_grads = T.grad(self.loss, self.params)
        all_grads = [T.clip(g, -5, 5) for g in all_grads]
        all_grads, norm = lasagne.updates.total_norm_constraint(
            all_grads, MAX_GRAD_NORM, return_norm=True)
        
        updates = lasagne.updates.adam(all_grads, self.params, learning_rate=0.001)
        
        self.train_fn = theano.function([self.q_var, self.answer_var, self.answer_mask, self.answer_inp_var],
                                  [self.loss, output, norm],
                                  updates=updates
                                 )
        
        self.test_fn = theano.function([self.q_var, self.answer_var, self.answer_mask, self.answer_inp_var], [self.loss, output, norm])

        self.pred_fn = theano.function([self.q_var, self.answer_inp_var], [output])
    def _empty_word_vector(self):
        return np.zeros((self.word_vector_size,), dtype=floatX)
    
    def _empty_inp_cnn_vector(self):
        return np.zeros((self.cnn_dim,), dtype=floatX)
    
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

        questions = np.zeros((self.batch_size, self.cnn_dim), dtype = 'float32')
        answers_inp = np.zeros((self.batch_size, self.SEQUENCE_LENGTH -1), dtype = 'int32')
        answers = np.zeros((self.batch_size, self.SEQUENCE_LENGTH), dtype = 'int32')
        mask = np.zeros((self.batch_size, self.SEQUENCE_LENGTH), dtype ='bool')
        img_ids = []

        idx = 0
        for slid, sid in zip(slids,stories):
            anno = split_dict_story[sid]
            input_anno = anno[slid]
            img_id = input_anno[1][0]
            img_ids.append(img_id)
            questions[idx,:] = features[fns_dict[img_id]]

            ans_idx = 0
            answers_inp[idx,0] = self.vocab_size
            answers[idx,0] = self.vocab_size
            mask[idx, 0] = True
            
            for word in input_anno[1][2]:
                ans_idx += 1
                if ans_idx >= self.SEQUENCE_LENGTH -1:
                    break
                answers_inp[idx, ans_idx] = word
                answers[idx, ans_idx] = word
                mask[idx, ans_idx] = True
                
            idx += 1
        return questions, answers, answers_inp, mask, img_ids
   
    
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
        anno_fn = os.path.join(split_dir,'annotions_filtered_fixed.txt')
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
                assert fea.shape[0] == fns.shape[0], "Should not happen, we have re-runed the feature extraction."
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
        
        q, ans, ans_inp, ans_mask, img_ids = self._process_batch_sind(batch_index, mode)
       
        
        ret = theano_fn(q, ans, ans_mask, ans_inp)
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"prediction": ret[1],
                "answers": ans,
                "current_loss": ret[0],
                "current_norm": ret[2],
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
                    if idx_prev[-1] == self.vocab['.']:
                        # This is the end.
                        beam_c[i].append(b)
                        continue

                    idx_prevs[i].append( idx_prev )
                    idx_of_idx[i].append(k)
                    idx_of_idx_len.append( len(idx_prev))
                    cnt_ins += 1

                    if len(idx_prev) > max_b:
                        max_b = len(idx_prev)

            if cnt_ins == 0:
                break
            
            x_i = np.zeros((cnt_ins, self.SEQUENCE_LENGTH-1), dtype = 'int32')
            q_i = np.zeros((cnt_ins, self.cnn_dim), dtype='float32')

            idx_base = 0
            for j,idx_prev_j in enumerate(idx_prevs):
                for m, idx_prev in enumerate(idx_prev_j):
                    for k in range(len(idx_prev)):
                        x_i[m + idx_base,k] = idx_prev[k]
                q_i[idx_base:idx_base + len(idx_prev_j),:] = q[j,:]

                idx_base += len(idx_prev_j)
            # This is really pain full.
            # Since the batch_size is fixed when creating the module. Thus,
            # we need to make them equal to the batch_size.
            pred = np.zeros((cnt_ins, x_i.shape[1]), dtype = 'float32')
            for i in range(0, x_i.shape[0], batch_size):
                start_idx = i
                end_idx = i + batch_size
                if end_idx > x_i.shape[0]:
                    end_idx = x_i.shape[0]
                    start_idx = end_idx - batch_size
                
                t = theano_fn(q_i[start_idx:end_idx,:], x_i[start_idx:end_idx,:])
                pred[start_idx:end_idx,:] = t[0]

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
            if nsteps >= 20:
                break
        for beams in batch_of_beams:
            pred = [(b[0], b[1]) for b in beams ]
            captions.append(pred)

        return {'captions':captions,
                'img_ids': img_ids}
