from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI
import torch

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 500
K = 5
p_cts = 20
p_bin = 0
var_cts = 1.
missing_mean = -1.00 #-3 ~ 7%, -2 ~ 15%, -1 ~ 30% , 0. ~ 50%
alpha_mean = -1.85
seed = 0

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_mean=missing_mean,
    alpha_mean=alpha_mean, seed=seed, mnar_sparsity=0.3
)

mnar = True
mnar = False
train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar, test=True)


# self = MLE(K, N, p_cts, p_bin, mnar=mnar)
self = ADVI(K, N, p_cts, p_bin, mnar=mnar)
# self = VIMC(K, N, p_cts, p_bin, mnar=mnar)
if mnar:
    B0 = torch.cat([B0, C0], 1)
    B = torch.cat([B, C], 1)
init = {"positions": Z, "heterogeneity": alpha, "bias": B0, "weight": B}
init = {k: v * (0.99 + 0.01*torch.rand_like(v)) for k, v in init.items()}
self.init(**init)

self.fit_path(train, test, reg=[10., 5., 2., 1., 0.5, 0.2, 0.1], max_iter=200, lr=0.01,
              init=init, Z_true=Z)


B.T
self.model.covariate_model.weight


for n, p in self.model.named_parameters():
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