from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI, VIMC, MLE, MICE
import torch
import numpy as np
from statsmodels.multivariate.factor import Factor
from NAIVI.utils.metrics import invariant_distance
# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 200
K = 2
p_cts = 5
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

mnar = False
train = JointDataset(i0, i1, None, X_cts, X_bin, return_missingness=mnar)
test = JointDataset(i0, i1, None, X_cts_missing, X_bin_missing, return_missingness=mnar, test=True)

if mnar:
    B0 = torch.cat([B0, C0], 1)
    B = torch.cat([B, C], 1)
init = {"positions": Z, "heterogeneity": alpha, "bias": B0, "weight": B}
# self = MLE(K, N, p_cts, p_bin, mnar=mnar)
self = ADVI(K, N, p_cts, p_bin, mnar=mnar)
# self = VIMC(K, N, p_cts, p_bin, mnar=mnar, n_samples=10)
# init = {k: v * (0.9 + 0.2*torch.rand_like(v)) for k, v in init.items()}
self.init(**init)

out = self.fit(train, test, reg=0., max_iter=200, lr=0.1, Z_true=Z, alpha_true=alpha)


x = self.covariate_weight / (self.model.covariate_model.model_cts.log_var.T / 2.).exp()
# x = self.covariate_weight / (self.model.covariate_model.cts_logvar.T / 2.).exp()
x.T @ x
B @ B.T


X = train[:][4]
fa = Factor(X.detach().numpy(), K, method="ml").fit()
fa.rotate("varimax")
fa.summary()



# compare BtB + Psi
x_cov_sample = np.cov(X.detach().numpy().T)
sds = np.sqrt(np.diag(x_cov_sample))

B_true = B.detach().numpy()
Psi_true = np.eye(p_cts) * var_cts
x_cov_true = B_true.T @ B_true + Psi_true

B_advi = self.covariate_weight.cpu().detach().numpy().T
Psi_advi = np.diag(self.model.covariate_model.model_cts.log_var.exp().cpu().detach().numpy().flatten())
x_cov_advi = B_advi.T @ B_advi + Psi_advi

B_fa = fa.loadings.T * sds.reshape((1, -1))
Psi_fa = np.diag(fa.uniqueness * sds ** 2)
x_cov_fa = B_fa.T @ B_fa + Psi_fa

print(np.round(x_cov_true, 2))
print(np.round(x_cov_sample, 2))
print(np.round(x_cov_advi, 2))
print(np.round(x_cov_fa, 2))

# compare BtB
BtB_true = B_true.T @ B_true
BtB_advi = B_advi.T @ B_advi
BtB_fa = B_fa.T @ B_fa
print(np.round(BtB_true, 2))
print(np.round(BtB_advi, 2))
print(np.round(BtB_fa, 2))

# compare scores
Z_fa = torch.Tensor(fa.factor_scoring(method="regression")).cuda()
Z_advi = self.latent_positions()
Z_true = Z.cuda()
print(invariant_distance(Z_true, Z_fa))
print(invariant_distance(Z_true, Z_advi))

# compute true posterior and best KL approximation
Psiinv = torch.eye(p_cts) / var_cts
vcov = B @ Psiinv @ B.T + torch.eye(K)
vcov = torch.inverse(vcov)
mean = (vcov @ B @ Psiinv @ (X - B0).T).T
diagvcov = torch.diag(vcov)

# get exact posterior from fa and its best KL approximation
Psiinv_fa = np.diag(1. / (fa.uniqueness * sds ** 2))
vcov_fa = B_fa @ Psiinv_fa @ B_fa.T + np.eye(K)
vcov_fa = np.linalg.inv(vcov_fa)
Xc = X - X.mean(0)
mean_fa = (vcov_fa @ B_fa @ Psiinv_fa @ (Xc.numpy()).T).T
diagvcov_fa = np.diag(vcov_fa)

# get

