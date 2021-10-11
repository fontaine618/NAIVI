from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI, VIMC, MLE, MICE, MCMC
import torch
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

import stan
import arviz as az
import matplotlib.pyplot as plt
plt.style.use("seaborn")

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 50
K = 5
p_cts = 10
p_bin = 0
var_cts = 1.
missing_mean = -1000.
alpha_mean = -2.
seed = 0
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

advi = VIMC(K, N, p_cts, p_bin, mnar=mnar, n_samples=1)
advi.fit(train, train, Z, batch_size=len(train),
         eps=5.e-6, max_iter=200, lr=0.1, alpha_true=alpha)

Z_advi = advi.latent_positions()
alpha_advi = advi.latent_heterogeneity()
ZZt_advi = (Z_advi @ Z_advi.T).detach().cpu().numpy()
A_logit = alpha_advi[i0] + alpha_advi[i1] + torch.sum(Z_advi[i0, :] * Z_advi[i1, :], 1, keepdim=True)
proba_advi = torch.sigmoid(A_logit).detach().cpu().numpy()

B0_advi = advi.model.covariate_model.bias
B_advi = advi.model.covariate_model.weight.T

Theta_X_advi = (B0_advi + torch.matmul(Z_advi, B_advi))[:, :p_cts].detach().cpu().numpy()
# -----------------------------------------------------------------------------
# MCMC
# -----------------------------------------------------------------------------

mcmc = MCMC(K, N, p_cts, p_bin, (0., 1.), (-2., 1.))
mcmc.fit(train, max_iter=1000, Z_true=Z.detach().cpu().numpy())

self = mcmc

ZZt_mcmc = mcmc.posterior_mean("ZZt")
proba_mcmc = mcmc.posterior_mean("proba").reshape((-1, 1))
Theta_X_mcmc = mcmc.posterior_mean("Theta_X")[:, :p_cts]

((ZZt_mcmc - ZZt_true)**2).sum() / (ZZt_true**2).sum()
((ZZt_advi - ZZt_true)**2).sum() / (ZZt_true**2).sum()

((proba_mcmc - proba_true)**2).sum() / (proba_true**2).sum()
((proba_advi - proba_true)**2).sum() / (proba_true**2).sum()

((Theta_X_mcmc - Theta_X_true)**2).sum() / (ZZt_true**2).sum()
((Theta_X_advi - Theta_X_true)**2).sum() / (ZZt_true**2).sum()



# # posterior predictive check
# Theta_X_post = self._fit.get("Theta_X").mean(2)
# Theta_X_post = Theta_X_post[:, :p_cts]
# Theta_X = (Z.detach().cpu().numpy() @ B.detach().cpu().numpy())[:, :p_cts]
# df = pd.DataFrame({
#     "post": Theta_X_post.flatten(),
#     "obs": Theta_X.flatten()
# })
# plt.scatter(df.post, df.obs)
# plt.show()
# np.round(self._fit.get("Theta_X")[:, :, -1], 1)
# np.round(Z.detach().cpu().numpy() @ B.detach().cpu().numpy(), 1)
# ax = az.plot_trace(self._fit, var_names=["B0", "B", "Z"])
# plt.show()
