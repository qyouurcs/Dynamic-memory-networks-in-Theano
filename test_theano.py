import theano
import theano.tensor as T
import numpy as np
import nn_utils


x = T.tensor3('x')

y = x[::-1,:,:]

f = theano.function(inputs = [x], outputs = y)

a = np.random.rand(2,3,2).astype('float32')

b = f(a)

print a[:,:,0]
print '\n'
print b[:,:,0]
#e_x = T.fvector('e_x')
#
#e_x2 = T.set_subtensor(e_x[0],0)
#
#
#
#f = theano.function(inputs = [e_x], outputs = e_x2)
#
#
#a = np.random.rand(5,).astype('float32')
#
#print a
#print f(a)
#
#ex = T.fmatrix('ex')
#
#def step(e_x):
#    e_x2 = T.set_subtensor(e_x[0],0)
#    return e_x, e_x2
#
#r,_ = theano.scan(fn = step, sequences = [ex])
#
#f2 = theano.function(inputs = [ex], outputs = r)
#
#x = np.random.rand(5,4).astype('float32')
#
#print x
#
#print
#print f2(x)[0]
#print
#
#print f2(x)[1]


#input_var = T.matrix('input_var') # (batch_size, seq_len, cnn_dim)
#att_mask = T.ivector('att_matrix')
#
#prob_sm = nn_utils.softmax_(input_var)
#loss_vec = T.nnet.categorical_crossentropy(prob_sm, att_mask)
#
#sf = theano.function(inputs = [input_var, att_mask], outputs = loss_vec)
#
#inp = np.random.rand(6,5).astype('float32')
#mask = np.random.rand(6,)
#mask[:] = 0
#mask = mask.astype('int32')
#
#t = sf(inp, mask)
#
#print inp
#print mask
#print 't', t.shape
#print t
#
