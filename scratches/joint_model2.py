import tensorflow as tf

from models.vmp.joint_model2 import JointModel2

tf.random.set_seed(1)
# problem dimension
N = 100
K = 3
p_cts = 2
p_bin = 2
p = p_cts + p_bin
var_adj = 1.
var_cov = 1.
missing_rate = 0.0

#-------------------------------------------------
# latent variables
Z = tf.random.normal((N, K), 0.0, 1.0, tf.float32)
alpha = tf.random.normal((N, 1), 0.0, 1.0, tf.float32)
# regression matrix
B = tf.ones((K, p)) * 2.
B0 = 1. * tf.ones((1, p))
# covariate model
Theta_X = tf.matmul(Z, B) + B0
# continuous variables
X_cts = tf.slice(Theta_X, [0, 0], [-1, p_cts])
X_cts += tf.random.normal((N, p_cts), 0.0, var_cov, tf.float32)
X_bin = tf.slice(Theta_X, [0, p_cts], [-1, -1])
X_bin = tf.random.normal((N, p_bin), 0.0, var_cov, tf.float32)
X_bin = tf.where(X_bin > 0., 1, 0)
# # missing values
# missing = tf.random.uniform(X_complete.shape) < missing_rate
# X = tf.where(missing, np.nan, X_complete)
# adjacency model
Theta_A = alpha + tf.transpose(alpha) + tf.matmul(Z, Z, transpose_b=True)
A = tf.where(Theta_A + tf.random.normal((N, N), 0., var_adj) > 0., 1, 0)

# ----------------------------------------------------
self = JointModel2(
    K=K,
    A=A,
    X_cts=X_cts,
    X_bin=X_bin
)

self.propagate(5)

print(Z)
print(self.positions.mean())

print(alpha)
print(self.heterogeneity.mean())

print(Theta_X)
print(self.covariate_mean.mean())

print(self.elbo())

for _ in range(100):
    # E Step
    self.propagate(10)
    # M Step
    for _ in range(10):
        self.compute_gradient()
        self.step()
        for k, v in self.parameters().items():
            print(k, v.value())
    print(self.elbo())


for k, v in self.gradient().items():
    print(k, v)

for k, v in self.parameters_value().items():
    print(k, v)

for k, v in self.factors().items():
    print(k, v)

with tf.GradientTape() as tape:
    target = self.elbo()
    for var in tape.watched_variables():
        print(var.name)

grad = g.gradient(target)