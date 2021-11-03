from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI, VIMC, MLE, MICE, MCMC, MAP
import torch
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

import arviz as az
import matplotlib.pyplot as plt
plt.style.use("seaborn")

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 200
K = 2
p_cts = 500
p_bin = 0
var_cts = 1.
missing_mean = -10000.
alpha_mean = -2.
seed = 4
mnar_sparsity = 1.0
mnar = False

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_mean=missing_mean,
    alpha_mean=alpha_mean, seed=seed, mnar_sparsity=mnar_sparsity
)

E = i0.shape[0]
p = p_bin + p_cts

train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)

ZZt_true = (Z @ Z.T).detach().cpu().numpy()
A_logit = alpha[i0] + alpha[i1] + torch.sum(Z[i0, :] * Z[i1, :], 1, keepdim=True)
proba_true = torch.sigmoid(A_logit).detach().cpu().numpy()

Theta_X_true = (B0 + torch.matmul(Z, B))[:, :p_cts].detach().cpu().numpy()
# -----------------------------------------------------------------------------
# ADVI
# -----------------------------------------------------------------------------

advi = MLE(K, N, p_cts, p_bin, mnar=mnar)
_, logs_advi = advi.fit(train, train, Z, batch_size=len(train),
         eps=5.e-6, max_iter=200, lr=0.1, alpha_true=alpha, return_log=True)

# -----------------------------------------------------------------------------
# VIMC
# -----------------------------------------------------------------------------

vimc = VIMC(K, N, p_cts, p_bin, mnar=mnar, n_samples=50)
_, logs_vimc = vimc.fit(train, train, Z, batch_size=len(train),
         eps=5.e-6, max_iter=200, lr=0.1, alpha_true=alpha, return_log=True)

# -----------------------------------------------------------------------------
# MAP & MLE
# -----------------------------------------------------------------------------

mapfit = MAP(K, N, p_cts, p_bin, mnar=mnar)
_, logs_mapi = mapfit.fit(train, train, Z, batch_size=len(train),
         eps=5.e-6, max_iter=200, lr=0.1, alpha_true=alpha, return_log=True)


mlefit = MLE(K, N, p_cts, p_bin, mnar=mnar)
_, logs_mle = mlefit.fit(train, train, Z, batch_size=len(train),
         eps=5.e-6, max_iter=500, lr=0.1, alpha_true=alpha, return_log=True)

# -----------------------------------------------------------------------------
# MCMC
# -----------------------------------------------------------------------------

mcmc = MCMC(K, N, p_cts, p_bin, (0., 1.), (-2., 1.))
train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar, cuda=False)
mcmc.fit(train, max_iter=1000, Z_true=Z.detach().cpu().numpy(), num_chains=10)

az.summary(mcmc._fit, "ZZt")["r_hat"].quantile(np.linspace(0., 1., 11))

