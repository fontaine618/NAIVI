import torch
from NNVI.utils.gen_data import generate_dataset

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------
N = 5
K = 2
p_cts = 100
p_bin = 0
var_cts = 1.
missing_rate = 0.1
alpha_mean = -1.85

Z, a, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_rate=missing_rate,
    alpha_mean=alpha_mean, seed=1
)


# -----------------------------------------------------------------------------
# Encoder test
# -----------------------------------------------------------------------------
from NNVI.mle.encoder import Encoder, Select

self = Encoder(K, N)
indices = i0

self(i0)

self = Select(N, K)
self(i0)

# -----------------------------------------------------------------------------
# Fit Model
# -----------------------------------------------------------------------------

from NNVI.utils.data import JointDataset
train = JointDataset(i0, i1, A, X_cts, X_bin)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing)

from NNVI.mle.model import MLE
self = MLE(K, N, p_cts, p_bin)
self.fit(train, test, Z, batch_size=len(train), eps=1.e-5, max_iter=1000, lr=0.01)





# 592  | 7048.9775  0.9146     0.9208    | 0.0954     0.3897    | 8106.0229  1.1999     0.8518

from NNVI.vimc.model import VIMC
self = VIMC(K, N, p_cts, p_bin, position_prior=(0., 1.), heterogeneity_prior=(0., 1.))
self.fit(train, test, Z, batch_size=len(train), eps=1.e-5, max_iter=1000, lr=0.01, n_sample=10)



torch.cat([self.latent_positions(), Z], 1)
torch.cat([self.latent_heterogeneity(), a], 1)

self.model.covariate_model.mean_model.bias
B0
torch.exp(self.model.covariate_model.cts_logvar)
var_cts

y_true = Z
y_pred = self.latent_positions()



