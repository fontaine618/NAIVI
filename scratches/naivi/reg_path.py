from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI, VIMC, MLE, MICE
import torch
import numpy as np

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 500
K = 5
p_cts = 100
p_bin = 0
var_cts = 1.
missing_mean = -100. #-3 ~ 7%, -2 ~ 15%, -1 ~ 30% , 0. ~ 50%
alpha_mean = -1.85
seed = 0
mnar_sparsity = 0.5

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_mean=missing_mean,
    alpha_mean=alpha_mean, seed=seed, mnar_sparsity=mnar_sparsity
)

torch.isnan(X_cts).double().mean()


# mnar = False
# train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
# test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar, test=True)
# self = MICE(K, N, p_cts, p_bin)
# out = self.fit(train, test)

mnar = True
# mnar = False
train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar, test=True)


if mnar:
    B0 = torch.cat([B0, C0], 1)
    B = torch.cat([B, C], 1)
init = {"positions": Z, "heterogeneity": alpha, "bias": B0, "weight": B}
# self = MLE(K, N, p_cts, p_bin, mnar=mnar)
self = ADVI(K, N, p_cts, p_bin, mnar=mnar)
# self = VIMC(K, N, p_cts, p_bin, mnar=mnar)
# init = {k: v * (0.9 + 0.2*torch.rand_like(v)) for k, v in init.items()}
self.init(**init)

out = self.fit_path(train, test, reg=10**np.linspace(1., -1., 11), max_iter=100, lr=0.1,
              init=init, Z_true=Z, alpha_true=alpha)


B.T
self.model.covariate_model.weight

step = "M"
self.select_params_for_step(step)
for n, p in self.model.named_parameters():
    if p.grad is not None:
        print(n)
    if p.requires_grad:
        print(n, p)
    if p.grad is not None:
        print(n, p.grad.norm("fro").item())


# reg=[0.1, 0.2]
# n_folds=5
# cv_seed=0
# fold=0
#
#
#
# reg=10**np.linspace(-1., -3., 20)
# self.cv_path(train, reg=reg, lr=0.01, max_iter=200)
#
# import pandas as pd
# cv_results = pd.DataFrame([(r, i, *self.cv_fit_[i][r]) for i in range(n_folds) for r in reg])
# cv_results.columns = ["reg", "fold", "iter", "grad norm", "train loss", "train mse", "train auroc",
#                       "inv.", "proj.", "test loss", "test mse", "test auroc", "non zero"]
# summary = cv_results.groupby("reg").agg("mean")
# summary["test loss"]