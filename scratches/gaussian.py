import tensorflow as tf
import sys
sys.path.append("/home/simon/Documents/NNVI/NNVI")
from models.distributions.gaussianarray import GaussianArray
from NNVI.vmp.vmp.vmp_factors import Sum, WeightedSum, Probit

factor = Sum()

x = GaussianArray.from_array(tf.random.normal((3, 2, 3), 0.0, 1.0), tf.random.normal((3, 2, 3), 0.0, 1.0) ** 2)

sum = factor.to_sum(x)

new_x = factor.to_x(x, sum)

x = GaussianArray.from_array(tf.random.normal((3, 2), 0.0, 1.0), tf.random.normal((3, 2), 0.0, 1.0) ** 2)
B = tf.random.normal((2, 5), 0.0, 1.0)
B0 = tf.random.normal((1, 5), -1.0, 1.0)

factor = WeightedSum()

result = factor.to_result(x, B, B0)

new_x = factor.to_x(x, result, B, B0)

m = tf.tensordot(x.mean(), B, 1) + B0
v = tf.tensordot(x.variance(), B**2, 1)
result = GaussianArray.from_array(m, v)


x = GaussianArray.from_array(tf.random.normal((3, 3), 0.0, 1.0), tf.random.normal((3, 3), 0.0, 1.0) ** 2)
A = tf.random.stateless_binomial((3, 3), (3, 3), 1, 0.2)

factor = Probit()
new_x = factor.to_x(x, A)

N = 10
K = 3