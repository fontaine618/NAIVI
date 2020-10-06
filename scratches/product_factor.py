import tensorflow as tf
import numpy as np
from models.distributions.gaussianarray import GaussianArray

from models.vmp.vmp_factors2 import Product
N = 3
K = 1
parent = GaussianArray.from_shape((N, K), 1.414, 1.)

upper = tf.linalg.band_part(tf.ones((N, N)), -1, 0) == 0
mean = tf.where(tf.expand_dims(upper, 2), 2., 0.)
variance = tf.where(tf.expand_dims(upper, 2), 5., np.inf)

child = GaussianArray.from_array(mean, variance)

self = Product(child, parent)


print(parent)
self.forward()
self.backward()
print(parent)

print(child)
self.to_child()
print(child.mean())

self.to_elbo()
