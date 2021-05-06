from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI
from NAIVI import MLE
from NAIVI import VIMC
from NAIVI import MICE
import torch
import numpy as np


# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 500
K = 5
p_cts = 10
p_bin = 0
var_cts = 1.
missing_mean = -1.00 #-3 ~ 7%, -2 ~ 15%, -1 ~ 30% , 0. ~ 50%
alpha_mean = -1.85
seed = 0

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_mean=missing_mean,
    alpha_mean=alpha_mean, seed=seed, mnar_sparsity=0.3
)

torch.isnan(X_cts).double().mean()

mnar = True
train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
self = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing,
                    return_missingness=mnar, test=True)
if mnar:
    B0 = torch.cat([B0, B0*0.001], 1)
    B = torch.cat([B, B*0.001], 1)

init = {"positions": Z, "heterogeneity": alpha, "bias": B0, "weight": B}
init = {k: v * (0.8 + 0.2*torch.rand_like(v)) for k, v in init.items()}
# -----------------------------------------------------------------------------
# VIMC fit
# -----------------------------------------------------------------------------

self = VIMC(K, N, p_cts, p_bin, n_samples=1, mnar=mnar)
self.init(**init)
self.fit(train, test, Z, batch_size=len(train),
         eps=5.e-6, max_iter=200, lr=0.1, reg=0.00047)

# -----------------------------------------------------------------------------
# ADVI fit
# -----------------------------------------------------------------------------

self = ADVI(K, N, p_cts, p_bin, mnar=mnar)
self.init(**init)
self.fit(train, test, Z, batch_size=len(train),
         eps=5.e-6, max_iter=200, lr=0.1, reg=0.000001)

# -----------------------------------------------------------------------------
# MLE fit
# -----------------------------------------------------------------------------

self = MLE(K, N, p_cts, p_bin, mnar=mnar)
self.init(**init)
self.fit(train, test, Z, batch_size=len(train),
         eps=5.e-6, max_iter=200, lr=0.1, reg=0.0013)

self.fit_path(train, test, Z, batch_size=len(train),
         eps=5.e-6, max_iter=20, lr=0.1, reg=10**np.linspace(-2.9, -3.1, 10))

# -----------------------------------------------------------------------------
# MICE fit
# -----------------------------------------------------------------------------

self = MICE(K, N, p_cts, p_bin)
self.fit(train, test)


self.model.covariate_model.weight.norm("fro", 1)
B.norm("fro", 0)
C.norm("fro", 0)


