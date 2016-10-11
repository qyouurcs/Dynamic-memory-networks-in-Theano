import theano
import theano.tensor as T
import numpy as np
import nn_utils

input_var = T.tensor3('input_var') # (batch_size, seq_len, cnn_dim)

input_var_rhp = T.reshape(input_var, (input_var.shape[0] * input_var.shape[1], input_var.shape[2]))

prob = nn_utils.softmax(input_var_rhp)
prob_rhp = T.reshape(prob, (input_var.shape[0], input_var.shape[1], input_var.shape[2]))

sf = theano.function(inputs = [input_var], outputs = prob_rhp)

inp = np.random.rand(4,5,3).astype('float32')

t = sf(inp)

print 't', t
print 'sum(2)', t.sum(2)
print 'sum(1)', t.sum(1)
print 'sum(0)', t.sum(0)

print

t2 = np.reshape(t, (t.shape[0] * t.shape[1], t.shape[2]))

print 't2, sum(0)'
print t2.sum(0)

print 't2, sum(1)'
print t2.sum(1)
