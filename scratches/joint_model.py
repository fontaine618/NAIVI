import tensorflow as tf
import tensorflow_probability as tfp
from NNVI.models.vmp import JointModel
from NNVI.models.vmp.vmp_factors import Prior, Product, Probit, Sum, AddVariance, Concatenate, WeightedSum, GaussianComparison
from NNVI.models.gaussian import GaussianArray


tf.random.set_seed(1)
# problem dimension
N = 10
K = 1
p = 1
var_adj = 1.
var_cov = 1.

#-------------------------------------------------
# latent variables
Z = tf.random.normal((N, K), 0.0, 1.0, tf.float32)
alpha = tf.random.normal((N, 1), 0.0, 1.0, tf.float32)
# regression matrix
B = tf.ones((K, p))
B0 = tf.ones((1, p))
# covariate model
Theta_X = tf.matmul(Z, B)
X = Theta_X + tf.random.normal((N, p), 0.0, var_cov, tf.float32)
# adjacency model
Theta_A = alpha + tf.transpose(alpha) + tf.matmul(Z, Z, transpose_b=True)
A = tf.where(Theta_A + tf.random.normal((N, N), 0., var_adj) > 0., 1, 0)

self = JointModel(N, K, A, X)

self._break_symmetry()
self.initialize_latent()

for _ in range(10):
    self.forward_adjacency()
    self.backward_adjacency()
    self.forward_covariate()
    self.backward_covariate()




tf.concat([Z, self.nodes["latent"].mean()], 1)
tf.concat([alpha, self.nodes["heterogeneity"].mean()], 1)

tf.concat([self.factors["weighted_sum"].message_to_result.mean(), X], 1)


# factor tests --------------------------------
# gaussian comparison
variance = tf.random.normal((1, p), 0., 1.) ** 2
self = GaussianComparison((N, p))
self.to_mean(X, variance)

# weighted sum
self = WeightedSum((N, K), (N, p))

x = GaussianArray.from_array(tf.random.normal((N, K), 0., 1.), tf.ones((N, K)))
result = GaussianArray.from_array(tf.random.normal((N, p), 0., 1.), 1. * tf.ones((N, p)))

self.to_result(x, B, B0)
self.to_x(x, result, B, B0)