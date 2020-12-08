import tensorflow as tf
import sys
sys.path.append("/home/simon/Documents/NNVI/NNVI")
from NNVI.vmp import Variational

model = Variational(6)


tf.random.set_seed(1)
# problem dimension
N = 20
K = 3
p = 7
# latent variables
Z = tf.random.normal((N, K), 0.0, 1.0, tf.float32)
alpha = tf.random.normal((N, 1), -1.0, 1.0, tf.float32)
# regression matrix
B = tf.random.normal((K, p), 0.0, 1.0, tf.float32)
# covariate model
Theta_X = tf.matmul(Z, B)
X = Theta_X + tf.random.normal((N, p), 0.0, 1.0, tf.float32)
# adjacency model
Theta_A = alpha + tf.transpose(alpha) + tf.matmul(Z, Z, transpose_b=True)
P_A = tf.nn.softmax(Theta_A)
A = tf.random.stateless_binomial(shape=(N, N), seed=(N, N), counts=1, probs=P_A, output_dtype=tf.dtypes.int32)