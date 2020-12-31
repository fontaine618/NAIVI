import tensorflow as tf
from models.vmp.logistic_utils import sigmoid_mean_and_variance, sigmoid_integrals

mean = tf.constant([-5., -1., 0., 1., 5.])
size = mean.shape
variance = tf.ones(size)

parameters = sigmoid_mean_and_variance(mean, variance)

integrals = sigmoid_integrals(mean, variance)

# simple MC comparison

def sigmoid(x):
    return 1. / (1. + tf.math.exp(-x))

N = 10000000
X = tf.random.normal(
    (*size, N),
    tf.tile(tf.expand_dims(mean, -1), [1, N]),
    tf.tile(tf.expand_dims(variance, -1), [1, N])
)
m = tf.reduce_mean(X * sigmoid(X), -1)
m /= integrals[0]
v = tf.reduce_mean((X-tf.expand_dims(m, -1)) ** 2 * sigmoid(X), -1)
v /= integrals[0]

print(m)
print(parameters[0])
print(v)
print(parameters[1])



# ------------------------------
# Logistic factor
import tensorflow as tf
import numpy as np

from models.distributions.gaussianarray import GaussianArray
from models.distributions.bernoulliarray import BernoulliArray
from models.vmp.vmp_factors2 import Logistic
shape = (5, 5)
parent = GaussianArray.from_array(
    tf.random.normal(shape, 0., 1.) ,
    tf.ones(shape)
)
A = tf.where(parent.mean() + tf.random.normal(shape) > 0., 1., 0.)

lower = tf.ones_like(A)
upper = tf.linalg.band_part(lower, -1, 0) == 0
A_lower = tf.where(upper, A, np.nan)


child = BernoulliArray.observed(A_lower)

self = Logistic(child, parent)

self.to_elbo()
self.to_child()
print(self.message_to_child)
self.to_parent()
print(parent)
print(self.message_to_parent)
print(self.predict())



# Probit
from models.vmp.compound_factors import NoisyProbit


self = NoisyProbit(child, parent, 1.)

self.forward()
print(self._probit.message_to_child)
print(parent)
self.backward()
print(child)
print(parent)
print(self.to_elbo())
print(self._noise.message_to_parent)