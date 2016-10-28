from mlp import *
import numpy as np

learning_rate=0.01
rng = numpy.random.RandomState(1234)

# may need to adjust size depending on your GPU
# but this is the value I am using
x = theano.shared(np.asarray(rng.uniform(size=(562411,429)),dtype='float32'))
y = theano.shared(np.asarray(rng.uniform(size=(562411,)),dtype='float32'))


classifier = MLP(rng=rng, input=x, n_in=429,
                     n_hidden=500, n_out=183)


cost = classifier.negative_log_likelihood(T.cast(y,'int32'))


gparams = []
for param in classifier.params:
    gparam = T.grad(cost, param)
    gparams.append(gparam)


updates = []
for param, gparam in zip(classifier.params, gparams):
    updates.append((param, param - learning_rate * gparam))


train_model = theano.function([],outputs=cost,updates=updates)
train_model()

