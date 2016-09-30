import theano
import theano.tensor as T
import numpy as np
import nn_utils

input_var = T.tensor3('input_var') # (batch_size, seq_len, cnn_dim)
dim = 40
cnn_dim = 4096

W_inp_emb_in = nn_utils.normal_param(std=0.1, shape=(dim, cnn_dim))
b_inp_emb_in = nn_utils.constant_param(value=0.0, shape=(dim,))

inp_var_shuffled = input_var.dimshuffle(1,2,0)

def _dot(x, W, b):
    return  T.dot(W, x) + b.dimshuffle(0, 'x')

inp_c_hist,_ = theano.scan(fn = _dot, sequences=inp_var_shuffled, non_sequences = [W_inp_emb_in, b_inp_emb_in])

f = theano.function(inputs = [input_var], outputs = inp_c_hist)

inp = np.random.rand(10,4,4096).astype('float32')

t = f(inp)
print t.shape
 
#k = T.iscalar("k")
#A = T.vector("A")
#
## Symbolic description of the result
#result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
#                                      outputs_info=T.ones_like(A),
#                                                                    non_sequences=A,
#                                                                                                  n_steps=k)
#
## We only care about A**k, but scan has provided us with A**1 through A**k.
## Discard the values that we don't care about. Scan is smart enough to
## notice this and not waste memory saving them.
#final_result = result
#
## compiled function that returns A**k
#power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

#print(power(range(10),2))
#print(power(range(10),4))


k = T.tensor3("K")

k2 = k.dimshuffle(1,2,0)

w = T.matrix('w')

m = T.dot(k, w)

t = T.tensor3('t')

mu = theano.function(inputs = [k, w], outputs = m)

def mu(a,b):
    return T.dot(b,a)

t,_ = theano.scan(fn = mu, sequences = [k2], non_sequences =[ w], outputs_info=None)

mu2 = theano.function(inputs = [k,w], outputs = t)

a = np.random.rand(4,3,10)
print a.flags

a = np.array(a, dtype = 'float32')
b = np.random.rand(5,10)

b= np.array(b, dtype= 'float32')

print mu2(a, b).shape


