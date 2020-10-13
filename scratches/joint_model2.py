import tensorflow as tf
import numpy as np

from models.vmp.joint_model2 import JointModel2

tf.random.set_seed(1)
# problem dimension
N = 100
K = 5
p_cts = 10
p_bin = 0
p = p_cts + p_bin
var_adj = 1.
var_cov = 1.
missing_rate = 0.1

#-------------------------------------------------
# latent variables
Z = tf.random.normal((N, K), 0.0, 1.0, tf.float32)
alpha = tf.random.normal((N, 1), -1.85, 0.5, tf.float32)
# regression matrix
B = tf.ones((K, p)) * 2.
B = tf.random.normal((K, p))
B0 = 1. * tf.ones((1, p))
# covariate model
Theta_X = tf.matmul(Z, B) + B0
# continuous variables
X_cts_all = tf.slice(Theta_X, [0, 0], [-1, p_cts])
X_cts_all += tf.random.normal((N, p_cts), 0.0, var_cov, tf.float32)
X_bin_all = tf.slice(Theta_X, [0, p_cts], [-1, -1])
X_bin_all += tf.random.normal((N, p_bin), 0.0, var_cov, tf.float32)
X_bin_all = tf.where(X_bin_all > 0., 1., 0.)
# # missing values
missing = tf.random.uniform(X_cts_all.shape) < missing_rate
X_cts = tf.where(missing, np.nan, X_cts_all)
X_cts_missing = tf.where(missing, X_cts_all, np.nan)
if p_cts == 0:
    X_cts_missing = None
missing = tf.random.uniform(X_bin_all.shape) < missing_rate
X_bin = tf.where(missing, np.nan, X_bin_all)
X_bin_missing = tf.where(missing, X_bin_all, np.nan)
if p_bin == 0:
    X_bin_missing = None
# adjacency model
Theta_A = alpha + tf.transpose(alpha) + tf.matmul(Z, Z, transpose_b=True)
P = 1. / (1. + tf.math.exp(- Theta_A))
A = tf.where(
    tf.random.uniform(P.shape) < P, 1., 0.
)
# Probit
# A = tf.where(Theta_A + tf.random.normal((N, N), 0., var_adj) > 0., 1., 0.)


upper = tf.linalg.band_part(tf.ones_like(A), -1, 0) == 0
A_lower = tf.where(upper, A, np.nan)
print(tf.reduce_sum(tf.where(tf.math.is_nan(A_lower), 0., A_lower)) * 2 / (N*(N-1)))
# A_lower = tf.ones_like(A) * np.nan

# ----------------------------------------------------
initial = {
        "bias": B0,
        "weights": B,
        "positions": Z,
        "heterogeneity": alpha
    }
self = JointModel2(
    K=K,
    A=A_lower,
    X_cts=X_cts,
    X_bin=X_bin,
    link_model="Logistic",
    bin_model="Logistic",
    initial=initial
)

self.fit(20, 5, 5, verbose=True,
         # X_cts_missing=X_cts_missing, X_bin_missing=X_bin_missing,
         # positions_true=Z
         )


tf.reduce_sum(tf.where(tf.math.is_nan(A_lower), 0., A_lower), 1)
tf.reduce_sum(tf.where(tf.math.is_nan(A_lower), 0., A_lower)) / N**2

self.predict_covariate()

for k, v in self.parameters_value().items():
    print(k)
    print(v)


Z
self.positions.mean()


print(Z)
print(self.positions.mean())

print(alpha)
print(self.heterogeneity.mean())
print(tf.concat([alpha, self.heterogeneity.mean()], 1) )

print(Theta_X)
print(self.covariate_mean.mean())

print(Theta_X - self.covariate_mean.mean())

self = self.inner_product_model._product
self.child.mean()[:,:,1]

tf.matmul(Z, Z, transpose_b=True)
m = self.positions.mean()
tf.matmul(m, m, transpose_b=True)


for name, node in self.nodes().items():
    print(name)
    print(node)




for _ in range(10):
    # E Step
    self.propagate(10)
    # M Step
    for _ in range(20):
        self.compute_gradient()
        self.step()
        # for k, v in self.parameters().items():
        #     print(k, v.value())
    print(self.to_elbo())


for k, v in self.gradient().items():
    print(k, v)



for k, v in self.factors().items():
    print(k, v)

with tf.GradientTape() as tape:
    target = self.to_elbo()
    for var in tape.watched_variables():
        print(var.name)

grad = g.gradient(target)