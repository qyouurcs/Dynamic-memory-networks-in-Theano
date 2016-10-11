import sys
import numpy as np
import sklearn.metrics as metrics
import argparse
import time
import json

import os

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
parser.add_argument('--dim', type=int, default=512, help='number of hidden units in input module GRU')
parser.add_argument('--cnn_dim', type=int, default=4096, help='number of hidden units in input module GRU')
parser.add_argument('--story_len', type=int, default=5, help='number of images in a story')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
parser.add_argument('--load_state', type=str, default="", help='state file path')
parser.add_argument('--answer_module', type=str, default="recurrent", help='answer module type: feedforward or recurrent')
parser.add_argument('--truncate_gradient', type=int, default=5, help='truncate_gradient')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate')

parser.add_argument('--mode', type=str, default="train", help='mode: train or test. Test mode required load_state')
parser.add_argument('--batch_size', type=int, default=80, help='no commment')
parser.add_argument('--data_dir', type=str, default="data/sind", help='data root directory')
parser.add_argument('--save_dir', type=str, default="states_rnn_show", help='data root directory')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--log_every', type=int, default=10, help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=1, help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (between 0 and 1)')
parser.set_defaults(shuffle=True)
args = parser.parse_args()

print args

network_name = args.prefix + '%s.n%d.bs%d%s%s%s' % (
    args.network, 
    args.dim, 
    args.batch_size, 
    "",
    "",
    (".d" + str(args.dropout)) if args.dropout>0 else ""
    )


#word2vec = utils.load_glove(args.word_vector_size)

args_dict = dict(args._get_kwargs())
#args_dict['word2vec'] = word2vec
args_dict['word2vec'] = None
    

# init class
if args.network == 'dmn_batch_sind':
    import rnn_batch_sind 
    dmn = rnn_batch_sind.DMN_batch(**args_dict)

else: 
    raise Exception("No such network known: " + args.network)
    

if args.load_state != "":
    dmn.load_state(args.load_state)

if not os.path.isdir(args.save_dir):
    os.makedirs(args.save_dir)

def do_epoch_beam(epoch, skipped=0):
    # mode is 'train' or 'test'
    batches_per_epoch = dmn.get_batches_per_epoch('test')
    
    all_candidates = [] 
    result_list = []
    json_fn = 't.json'
    #for i in range(0, batches_per_epoch):
    for i in range(0, 2):
        step_data = dmn.step_beam(i)
    
        for caption,img_id in zip(step_data['captions'], step_data['img_ids']):
            top_prediction = caption[0]
            # ix 0 is the END token, skip that
            candidate = ' '.join([ dmn.ivocab[ix] for ix in top_prediction[1] if ix != dmn.vocab['.'] and ix < len(dmn.vocab) ])
            pdb.set_trace()
            logging.info('loss: %f', caption[1])
            logging.info('candidate: %s', candidate)
            all_candidates.append(candidate)
            cur_img = {}
            cur_img['image_id'] = img_id
            cur_img['caption'] = candidate
            result_list.append(cur_img)
            # We need to calculate the BLEU score, thus we need to format the datas
            # the way did in neural talk, which has the perl codes for this.

    pdb.set_trace()
    with open(json_fn, 'w') as json_fid:
        json.dump(result_list, json_fid)


def do_epoch(mode, epoch, skipped=0):
    # mode is 'train' or 'test'
    y_true = []
    y_pred = []
    avg_loss = 0.0
    prev_time = time.time()
    
    batches_per_epoch = dmn.get_batches_per_epoch(mode)
    
    for i in range(0, batches_per_epoch):
        step_data = dmn.step(i, mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        current_skip = (step_data["skipped"] if "skipped" in step_data else 0)
        log = step_data["log"]
        
        skipped += current_skip
        
        if current_skip == 0:
            avg_loss += current_loss
            answers = np.reshape(answers, (answers.size,))
            preds = prediction.argmax(axis = 1)
            for x,y in zip(answers, preds):
                if x > 0:
                    y_true.append(x)
                    y_pred.append(y)
            
            #for x in prediction.argmax(axis=1):
            #    y_pred.append(x)
            
            # TODO: save the state sometimes
            if (i % args.log_every == 0):
                cur_time = time.time()
                print ("  %sing: %d %d / %d \t loss: %.3f \t avg_loss: %.3f \t skipped: %d \t %s \t time: %.2fs" % 
                    (mode, epoch, i * args.batch_size, batches_per_epoch * args.batch_size, 
                     current_loss, avg_loss / (i + 1), skipped, log, cur_time - prev_time))
                prev_time = cur_time
        
        if np.isnan(current_loss):
            print "==> current loss IS NaN. This should never happen :) " 
            exit()

    avg_loss /= batches_per_epoch
    print "\n  %s loss = %.5f" % (mode, avg_loss)
    print "confusion matrix:"
    print metrics.confusion_matrix(y_true, y_pred)
    
    accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    print "accuracy: %.2f percent" % (accuracy * 100.0 / len(y_true) )
    
    return avg_loss, skipped


if args.mode == 'train':
    print "==> training"   	
    skipped = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        
        if args.shuffle:
            dmn.shuffle_train_set()
        
        _, skipped = do_epoch('train', epoch, skipped)
        
        epoch_loss, skipped = do_epoch('test', epoch, skipped)
        
        state_name = '%s/%s.epoch%d.test%.5f.state' % (args.save_dir, network_name, epoch, epoch_loss)

        if (epoch % args.save_every == 0):    
            print "==> saving ... %s" % state_name
            dmn.save_params(state_name, epoch)
        
        print "epoch %d took %.3fs" % (epoch, float(time.time()) - start_time)

elif args.mode == 'test':
    file = open('last_tested_model.json', 'w+')
    data = dict(args._get_kwargs())
    data["id"] = network_name
    data["name"] = network_name
    data["description"] = ""
    data["vocab"] = dmn.vocab.keys()
    json.dump(data, file, indent=2)
    do_epoch('test', 0)
elif args.mode == 'test_beam':
    file = open('last_tested_model.json', 'w+')
    data = dict(args._get_kwargs())
    data["id"] = network_name
    data["name"] = network_name
    data["description"] = ""
    data["vocab"] = dmn.vocab.keys()
    json.dump(data, file, indent=2)
    do_epoch_beam('test', 0)


else:
    raise Exception("unknown mode")
