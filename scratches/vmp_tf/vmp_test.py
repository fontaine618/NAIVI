import torch
import tensorflow as tf
import numpy as np
# data
from NAIVI.utils.gen_data import generate_dataset
# VMP
from NAIVI.vmp_tf.vmp.joint_model2 import JointModel2
# MLE
from NAIVI.utils.data import JointDataset
from NAIVI.mle.model import MLE

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 100
K = 5
p_cts = 100
p_bin = 0
var_cts = 1.
missing_rate = 0.1
alpha_mean = -1.85

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_rate=missing_rate,
    alpha_mean=alpha_mean, seed=1
)

# -----------------------------------------------------------------------------
# MLE fit
# -----------------------------------------------------------------------------

train = JointDataset(i0, i1, A, X_cts, X_bin)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing)

self = MLE(K, N, p_cts, p_bin)
self.fit(train, test, Z, batch_size=len(train), eps=1.e-5, max_iter=1000, lr=0.01)

# -----------------------------------------------------------------------------
# VMP fit
# -----------------------------------------------------------------------------

# map to vmp_tf data structure

A_lower = torch.ones((N, N)) * np.nan
A_lower.index_put_((i0, i1), A.view(-1))

if X_bin is None:
    X_bin = torch.zeros((N, 0))
if X_cts is None:
    X_cts = torch.zeros((N, 0))
if X_bin_missing is None:
    X_bin_missing = torch.zeros((N, 0))
if X_cts_missing is None:
    X_cts_missing = torch.zeros((N, 0))

initial = {
        "bias": tf.convert_to_tensor(B0.numpy()),
        "weights": tf.convert_to_tensor(B.numpy()),
        "positions": tf.convert_to_tensor(Z.numpy()),
        "heterogeneity": tf.convert_to_tensor(alpha.numpy())
    }

self = JointModel2(
    K=K,
    A=tf.convert_to_tensor(A_lower.numpy(), dtype=np.float32),
    X_cts=tf.convert_to_tensor(X_cts.numpy(), dtype=np.float32),
    X_bin=tf.convert_to_tensor(X_bin.numpy(), dtype=np.float32),
    link_model="Logistic",
    bin_model="Logistic",
    initial=initial
)

self.set_lr(100.)

self.fit(10, 1, 0, verbose=True,
         X_cts_missing=tf.convert_to_tensor(X_cts_missing.numpy(), dtype=np.float32),
         X_bin_missing=tf.convert_to_tensor(X_bin_missing.numpy(), dtype=np.float32),
         positions_true=tf.convert_to_tensor(Z.numpy(), dtype=np.float32)
         )

self.fit(10, 0, 1, verbose=True,
         X_cts_missing=tf.convert_to_tensor(X_cts_missing.numpy(), dtype=np.float32),
         X_bin_missing=tf.convert_to_tensor(X_bin_missing.numpy(), dtype=np.float32),
         positions_true=tf.convert_to_tensor(Z.numpy(), dtype=np.float32)
         )

for n in self.parameters():
    print(n)

self.position_prior.mean.value()
self.position_prior.log_var.value()
self.heterogeneity_prior.mean.value()
self.heterogeneity_prior.log_var.value()
self.mean_model.bias.value()
self.mean_model.weight.value()

self.positions.mean()
self.heterogeneity.mean() - alpha



