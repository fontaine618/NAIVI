import tensorflow as tf
import numpy as np
from NNVI.vmp.vmp import JointModel
from NNVI.vmp.vmp.vmp_factors import WeightedSum, GaussianComparison
from models.distributions.gaussianarray import GaussianArray


tf.random.set_seed(1)
# problem dimension
N = 100
K = 1
p = 5
var_adj = 1.
var_cov = 1.
missing_rate = 0.1

#-------------------------------------------------
# latent variables
Z = tf.random.normal((N, K), 0.0, 1.0, tf.float32)
alpha = tf.random.normal((N, 1), 0.0, 1.0, tf.float32)
# regression matrix
B = tf.ones((K, p)) * -3.
B0 = +1. * tf.ones((1, p))
# covariate model
Theta_X = tf.matmul(Z, B) + B0
X_complete = Theta_X + tf.random.normal((N, p), 0.0, var_cov, tf.float32)
# missing values
missing = tf.random.uniform(X_complete.shape) < missing_rate
X = tf.where(missing, np.nan, X_complete)
# adjacency model
Theta_A = alpha + tf.transpose(alpha) + tf.matmul(Z, Z, transpose_b=True)
A = tf.where(Theta_A + tf.random.normal((N, N), 0., var_adj) > 0., 1, 0)

self = JointModel(K, A, X)

tf.random.set_seed(1)
self._break_symmetry()
self.initialize_latent()


# EM algorithm
lr = 0.01

for i in range(10):
    print("=" * 60)
    print("=== E-step")
    for j in range(10):
        print(i, j, self.pass_and_elbo().numpy())
    print("=== M-step")
    for j in range(10):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(self._parameters())
            target = self.elbo2()
        grad = g.gradient(target, self._parameters())
        for k, v in grad.items():
            self.parameters[k].grad = v
            self.parameters[k].step(lr * (1.0 if k == "noise_adjacency" else self.N))
        print(i, j, target.numpy())


for k, v in self.parameters.items():
    print(k)
    print(v.value())







for _ in range(10):
    self.pass_and_elbo()


with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
    g.watch(self._parameters())
    target = self.pass_and_elbo()

self.propagate()
with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
    g.watch(self._parameters())
    target = self.elbo2()


grad = g.gradient(target, self._parameters())
for k, v in self.parameters.items():
    print(k)
    print(grad[k])
    print(v.value())









lr = 0.01

for _ in range(30):

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
        g.watch(self._parameters())
        target = self.pass_and_elbo()

    grad = g.gradient(target, self._parameters())
    for k, v in grad.items():
        self.parameters[k].grad = v
        #self.parameters[k].step(lr)
        self.parameters[k].step(lr * (1.0 if k == "noise_adjacency" else self.N))
    print(target)


for k, v in self.parameters.items():
    print(k)
    print(grad[k])
    print(v.value())




for i in range(100):
    self.forward_adjacency()
    self.backward_adjacency()
    self.forward_covariate()
    self.backward_covariate()
    self.elbo()





tf.concat([Z, self.nodes["latent"].mean()], 1)
tf.concat([alpha, self.nodes["heterogeneity"].mean()], 1)

tf.concat([self.factors["weighted_sum"].message_to_result.mean(), X], 1)

self.factors["weighted_sum"].message_to_result.mean()
X_complete
missing

tf.where(missing, self.nodes["linear_predictor_covariate"].mean(), 0.)
tf.where(missing, self.factors["weighted_sum"].message_to_result.mean(), 0.)
tf.where(missing, X_complete, 0.)


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


# ---------------------------------------
# Missing values



GaussianArray.observed(X)


mean=self.nodes["linear_predictor_covariate"]
x=self.nodes["covariates_continuous"]
variance=self.parameters["noise_covariate"].value()


self = self.nodes["covariates_continuous"]


# --------------------------------------
# predict links
self.links_proba() >0.5
self.factors["adjacency"].to_result(self.nodes["noisy_linear_predictor_adjacency"])
A