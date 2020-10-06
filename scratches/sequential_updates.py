import tensorflow as tf
import numpy as np
from models.distributions.gaussianarray import GaussianArray
from models.distributions.bernoulliarray import BernoulliArray


# -----------------------------------------------------------------------------
# WEIGHTED SUM
from models.vmp.vmp_factors2 import WeightedSum
n = 5
K = 7
p = 2

bias = tf.ones((1, p))
weights = tf.ones((K, p))

parent = GaussianArray.from_shape((n, K), 1., 2000000.)
child = GaussianArray.from_shape((n, p), 0., 1.)


parent = GaussianArray.from_shape((n, K), 1., 2.)
child = GaussianArray.from_shape((n, p), 5., 8.)

self = WeightedSum(child, parent, bias, weights)

print(parent)
self.to_child()
print(self.message_to_child)
self.to_parent()
print(self.message_to_parent)
print(parent)

print(child)
self.to_child()
print(child)

self.to_elbo()
