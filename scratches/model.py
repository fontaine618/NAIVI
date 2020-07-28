import tensorflow as tf
import tensorflow_probability as tfp
from NNVI.models.vmp import InnerProductLatentSpaceModel
from NNVI.models.vmp.vmp_factors import Prior, Product, Probit, Sum, AddVariance, Concatenate, WeightedSum
from NNVI.models.gaussianarray import GaussianArray


tf.random.set_seed(1)
# problem dimension
N = 100
K = 3
p = 4

#-------------------------------------------------
# latent variables
Z = tf.random.normal((N, K), 0.0, 1.0, tf.float32)
alpha = tf.random.normal((N, 1), 0.0, 1.0, tf.float32)
# regression matrix
B = tf.random.normal((K, p), 0.0, 1.0, tf.float32)
# covariate model
Theta_X = tf.matmul(Z, B)
X = Theta_X + tf.random.normal((N, p), 0.0, 1.0, tf.float32)
# adjacency model
Theta_A = alpha + tf.transpose(alpha) + tf.matmul(Z, Z, transpose_b=True)
# P_A = tf.nn.softmax(Theta_A)
# A = tf.random.stateless_binomial(shape=(N, N), seed=(N, N), counts=1, probs=P_A, output_dtype=tf.dtypes.int32)
A = tf.where(Theta_A + tf.random.normal((N, N), 0., 1.) > 0., 1, 0)

self = InnerProductLatentSpaceModel(N, K, A)

for _ in range(10):
    self.forward()
    self.backward()



tf.concat([Z, self.nodes["latent"].mean()], 1)
tf.concat([alpha, self.nodes["heterogeneity"].mean()], 1)


#
self.nodes["latent"]
self.nodes["heterogeneity"].mean() * alpha

tf.matmul(self.nodes["heterogeneity"].mean(), self.nodes["heterogeneity"].mean(), transpose_b=True)

self.factors["noise"].message_to_x.mean()
tf.reduce_sum(tf.where(self.nodes["noisy_linear_predictor"].mean() > 0., 1, 0) - A)
tf.reduce_sum(tf.where(self.factors["noise"].message_to_x.mean() > 0., 1, 0) - A)

x = self.nodes["vector"]
sum = self.nodes["linear_predictor"]


# factor tests --------------------------------
# sum
self = Sum((N, K+2), (N, ))
x = GaussianArray.from_array(tf.random.normal((N, K+2), 0., 1.), tf.ones((N, K+2)))
sum = GaussianArray.from_array(tf.random.normal((N, ), 0., 1.), 1.2*tf.ones((N, )))

self.to_x(x, sum)
self.to_sum(x)

print((self.message_to_x * x).mean())
print((self.message_to_sum * sum).mean())

# product
self = Product((N, K), (N, N, K))

x = GaussianArray.from_array(tf.random.normal((N, K), 0., 1.), tf.ones((N, K)))
product = GaussianArray.from_array(tf.random.normal((N, N, K), 0., 1.), 0.1 * tf.ones((N, N, K)))

self.to_x(x, product)
self.to_product(x)

print((self.message_to_x * x))
print((self.message_to_product * product))

# concatenate

self = Concatenate({"a_u": (N, N, 1), "a_v": (N, N, 1), "s_uv": (N, N, K)}, (N, N, K+2))
alpha = GaussianArray.from_array(tf.random.normal((N, 1), -1.0, 1.0, tf.float32), tf.ones((N, 1)))
a_u = GaussianArray(
    tf.tile(tf.expand_dims(alpha.precision(), 0), [N, 1, 1]),
    tf.tile(tf.expand_dims(alpha.mean_times_precision(), 0), [N, 1, 1])
)
a_v = GaussianArray(
    tf.tile(tf.expand_dims(alpha.precision(), 1), [1, N, 1]),
    tf.tile(tf.expand_dims(alpha.mean_times_precision(), 1), [1, N, 1])
)
s_uv = GaussianArray.from_array(tf.random.normal((N, N, K), 0., 1.), 0.1 * tf.ones((N, N, K)))
x = {"a_u": a_u, "a_v": a_v, "s_uv": s_uv}
v = GaussianArray.from_array(tf.random.normal((N, N, K+2), 0., 1.), 0.1 * tf.ones((N, N, K+2)))

self.to_v(x)
self.to_x(v)

# ----------

GaussianArray.from_shape((N, N, N+2), 0.0, 1.0)

self.factors["concatenate"].to_x(GaussianArray.from_shape((N, N, N+2), 0.0, 1.0))
for k, m in self.factors["concatenate"].message_to_x.items():
    print(m.shape())


self._update_latent_variable()



product = self.nodes["product"]
product.mean()
b = self.nodes["latent_variable"]
b.mean()



