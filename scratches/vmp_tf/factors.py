import tensorflow as tf
import numpy as np
from models.distributions.gaussianarray import GaussianArray
from models.distributions.bernoulliarray import BernoulliArray

# -----------------------------------------------------------------------------
# PRIOR
from models.vmp.vmp_factors2 import Prior
child = GaussianArray.uniform((3, 3))
self = Prior(child, 0., 1.)
self.forward()
self.to_elbo()

# -----------------------------------------------------------------------------
# ADD VARIANCE
from models.vmp.vmp_factors2 import AddVariance
parent = GaussianArray.from_shape((3, 3), 0., 1.)
child = GaussianArray.from_shape((3, 3), 0., 1.)
self = AddVariance(child, parent, 1.)

self.to_child()
print(child)
self.to_parent()
print(parent)
self.to_elbo()

# -----------------------------------------------------------------------------
# PROBIT
from models.vmp.vmp_factors2 import Probit
parent = GaussianArray.from_shape((5, 5), 0., 1.)
A = tf.where(tf.random.normal((5, 5)) > 0., 1., 0.)

missing = tf.random.uniform(A.shape) < 0.2
A = tf.where(missing, np.nan, A)

child = BernoulliArray.observed(A)
self = Probit(child, parent)

self.to_elbo()
self.to_child()
print(self.message_to_child)
self.to_parent()
print(parent)
print(self.message_to_parent)

# -----------------------------------------------------------------------------
# NOISY PROBIT
from models.vmp.compound_factors import NoisyProbit, InnerProductModel, GLM

parent = GaussianArray.from_shape((5, 5), 0., 1.)
A = tf.where(tf.random.normal((5, 5)) > 0., 1., 0.)

missing = tf.random.uniform(A.shape) < 0.2
A = tf.where(missing, np.nan, A)

child = BernoulliArray.observed(A)
self = NoisyProbit(child, parent, 1.)

self.forward()
print(self._probit.message_to_child)
print(parent)
self.backward()
print(child)
print(parent)
print(self.to_elbo())

# -----------------------------------------------------------------------------
# GAUSSIAN COMPARISON
from models.vmp.vmp_factors2 import AddVariance
parent = GaussianArray.from_shape((3, 3), 0., 1.)
X = tf.random.normal((3, 3))
missing = tf.random.uniform(X.shape) < 0.5
X = tf.where(missing, np.nan, X)
child = GaussianArray.observed(X)

self = AddVariance(child, parent, 1.)

print(parent)
print(child)
self.to_child()
print(child)
self.to_parent()
print(parent)
self.to_elbo()


# -----------------------------------------------------------------------------
# SUM
from models.vmp.vmp_factors2 import Sum
parent = GaussianArray.from_shape((3, 2), 1., 100000.)
child = GaussianArray.from_shape((3, ), 3., 1.)
self = Sum(child, parent)


parent = GaussianArray.from_shape((3, 2), 1., 1.)
child = GaussianArray.from_shape((3, ), 3., 100000.)
self = Sum(child, parent)

print(parent)
self.to_child()
self.to_parent()
print(parent)

print(child)
self.to_child()
print(child)

self.to_elbo()

# -----------------------------------------------------------------------------
# PRODUCT
from models.vmp.vmp_factors2 import Product

parent = GaussianArray.from_shape((3, 2), 1.414, 1)
child = GaussianArray.from_shape((3, 3, 2), 2., 1.)

parent = GaussianArray.from_shape((3, 2), 1., 1.)
child = GaussianArray.from_shape((3, 3, 2), 2., 100000.)

self = Product(child, parent)

for _ in range(100):
    self.to_child()
    self.to_parent()
    print(parent)

print(child)
self.to_child()
print(child)

self.to_elbo()



# -----------------------------------------------------------------------------
# WEIGHTED SUM
from models.vmp.vmp_factors2 import WeightedSum
parent = GaussianArray.from_shape((3, 5), 1., 2000000.)
child = GaussianArray.from_shape((3, 2), 0., 1.)
self = WeightedSum(child, parent)


parent = GaussianArray.from_shape((3, 5), 1., 2.)
child = GaussianArray.from_shape((3, 2), 4., 8.)
self = WeightedSum(child, parent)


print(parent)
self.to_child()
self.to_parent()
print(parent)

print(child)
self.to_child()
print(child)

self.to_elbo()


# -----------------------------------------------------------------------------
# CONCATENATE
from models.vmp.vmp_factors2 import Concatenate
parts = {
    "A": GaussianArray.from_shape((3, 3), 3., 2.),
    "B": GaussianArray.from_shape((3, 2), 2., 2.),
    "C": GaussianArray.from_shape((3, 1), 1., 2.)
}
vector = GaussianArray.from_shape((3, 6), 0., 1.)
self = Concatenate(vector, parts)

print(vector)
print(parts)
self.to_vector()
print(vector)
self.to_parts()
print(parts)
self.to_elbo()





# -----------------------------------------------------------------------------
# EXPAND TRANSPOSE
from models.vmp.vmp_factors2 import ExpandTranspose
parent = GaussianArray.from_array(
    tf.random.normal((3, 1), 0., 1.),
    tf.ones((3, 1))*100000
)
child = GaussianArray.from_shape((3, 3, 2), 0., 1.)
self = ExpandTranspose(child, parent)

print(parent)
print(child)
self.to_child()
print(child)
self.to_parent()
print(parent)
self.to_elbo()



# -----------------------------------------------------------------------------
# INNER PRODUCT MODEL
N = 3
K = 2
positions = GaussianArray.from_array(
    tf.random.normal((N, K), 0., 1.),
    tf.ones((N, K)) * 100000.
)
heterogeneity = GaussianArray.from_array(
    tf.random.normal((N, 1), 0., 1.),
    tf.ones((N, 1)) * 100000.
)
linear_predictor = GaussianArray.from_array(
    tf.random.normal((N, N), 0., 1.),
    tf.ones((N, N))
)

self = InnerProductModel(positions, heterogeneity, linear_predictor)

self.forward()
print(positions)
print(heterogeneity)
print(linear_predictor)
self.backward()
print(positions)
print(heterogeneity)
print(linear_predictor)


# -----------------------------------------------------------------------------
# WEIGHTED SUM

parent = GaussianArray.from_shape((3, 2), 0., 2.)
child_cts = GaussianArray.observed(tf.random.normal((3, 1), 0., 1.))
A = tf.where(tf.random.normal((3, 1)) > 0., 1., 0.)
child_bin = BernoulliArray.observed(A)
self = GLM(
    parent,
    child_cts,
    child_bin
)

print(parent)
self.forward()
self.backward()
print(parent)