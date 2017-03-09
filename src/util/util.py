from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import random

default_rng = np.random.RandomState(100)
default_mrng = MRG_RandomStreams(default_rng.randint(9999))
default_srng = default_mrng

ReLU = lambda x: x * (x > 0)
sigmoid = T.nnet.sigmoid
tanh = T.tanh
softmax = T.nnet.softmax
linear = lambda x: x

def get_activation_by_name(name):
    if name.lower() == "relu":
        return ReLU
    elif name.lower() == "sigmoid":
        return sigmoid
    elif name.lower() == "tanh":
        return tanh
    elif name.lower() == "softmax":
        return softmax
    elif name.lower() == "none" or name.lower() == "linear":
        return linear
    else:
        raise Exception("unknown activation type: {}".format(name) )


def set_default_rng_seed(seed):
    global default_rng, default_srng
    random.seed(seed)
    default_rng = np.random.RandomState(random.randint(0,9999))
    default_srng = T.shared_randomstreams.RandomStreams(default_rng.randint(9999))

def random_init(size, rng=None, rng_type=None):
    if rng is None: rng = default_rng
    if rng_type is None:
        #vals = rng.standard_normal(size)
        vals = rng.uniform(low=-0.25, high=0.25, size=size)

    elif rng_type == "normal":
        vals = rng.standard_normal(size)

    elif rng_type == "uniform":
        vals = rng.uniform(low=-3.0**0.5, high=3.0**0.5, size=size)

    elif rng_type == "xavier":
    	dim = (np.sum(size) + 1)**0.5
        vals = rng.uniform(low=-6**0.5/dim, high=6**0.5/dim, size=size)
    
    else:
        raise Exception("unknown random inittype: {}".format(rng_type))

    return vals.astype(theano.config.floatX)

def create_shared(vals, name=None):
	return theano.shared(vals, name=name)
	
