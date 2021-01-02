import torch
from NNVI.utils.gen_data import generate_dataset
from NNVI.utils.data import JointDataset
from NNVI.advi.model import ADVI
from NNVI.mle.model import MLE
from NNVI.vimc.model import VIMC

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 500
K = 5
p_cts = 10
p_bin = 0
var_cts = 1.
missing_rate = 0.1
alpha_mean = -1.85

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_rate=missing_rate,
    alpha_mean=alpha_mean, seed=1
)

train = JointDataset(i0, i1, A, X_cts, X_bin)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing)


# -----------------------------------------------------------------------------
# VIMC fit
# -----------------------------------------------------------------------------

self = VIMC(K, N, p_cts, p_bin)
self.init(positions=Z.cuda(), heterogeneity=alpha.cuda())
self.fit(train, test, Z.cuda(), batch_size=len(train), eps=1.e-6, max_iter=200, lr=0.01, n_sample=1)

# -----------------------------------------------------------------------------
# ADVI fit
# -----------------------------------------------------------------------------

self = ADVI(K, N, p_cts, p_bin)
self.init(positions=Z.cuda(), heterogeneity=alpha.cuda())
self.fit(train, test, Z, batch_size=len(train), eps=1.e-6, max_iter=200, lr=0.01)

# -----------------------------------------------------------------------------
# MLE fit
# -----------------------------------------------------------------------------

self = MLE(K, N, p_cts, p_bin)
self.init(positions=Z.cuda(), heterogeneity=alpha.cuda())
self.fit(train, test, Z, batch_size=len(train), eps=1.e-6, max_iter=200, lr=0.1)

