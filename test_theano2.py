import theano
import theano.tensor as T
import numpy as np
import nn_utils
import pdb

input_var = T.matrix('input_var') # (batch_size, seq_len, cnn_dim)

input_var_rhp = T.repeat(input_var, 4, 0)
input_var_rhp2 = T.tile(input_var, (5, 1))

sf = theano.function(inputs = [input_var], outputs = input_var_rhp)
sf2 = theano.function(inputs = [input_var], outputs = input_var_rhp2)

inp = np.random.rand(2,3).astype('float32')

print inp
t = sf(inp)
t2 = sf2(inp)

pdb.set_trace()
print t

