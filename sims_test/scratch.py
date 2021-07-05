from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI, VIMC, MLE, MICE
import torch
import numpy as np

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 100
K = 5
p_cts = 100
p_bin = 0
var_cts = 1.
missing_mean = -0.5 #-3 ~ 7%, -2 ~ 15%, -1 ~ 30% , 0. ~ 50%
alpha_mean = -1.85
seed = 0
mnar_sparsity = 0.1

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_mean=missing_mean,
    alpha_mean=alpha_mean, seed=seed, mnar_sparsity=mnar_sparsity
)

mnar = True
# mnar = False
train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar, test=True)


if mnar:
    BC0 = torch.cat([B0, C0], 1)
    BC = torch.cat([B, C], 1)
else:
	BC0 = B0
	BC = B
init = {"positions": Z, "heterogeneity": alpha, "bias": BC0, "weight": BC}
self = MLE(K, N, p_cts, p_bin, mnar=mnar)
# self = ADVI(K, N, p_cts, p_bin, mnar=mnar)
# self = VIMC(K, N, p_cts, p_bin, mnar=mnar)
# init = {k: v * (0.9 + 0.2*torch.rand_like(v)) for k, v in init.items()}
self.init(**init)

# out = self.fit(train, test, reg=1., max_iter=100, lr=0.1, Z_true=Z, alpha_true=alpha)

out = self.fit_path(train, test, reg=[10., 5., 2., 1., 0.5, 0.2, 0.1], max_iter=200, lr=0.1,
              init=init, Z_true=Z, alpha_true=alpha)

