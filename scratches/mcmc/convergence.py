from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI, VIMC, MLE, MICE, MCMC, MAP
import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from NAIVI_experiments.display import colormap, to_display
COLORS = colormap
DICT = to_display

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
# 50/100, 50/20, 200/500, 200/50
N = 50
K = 5
p_cts = 20
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

ZZt_true = Z @ Z.T
A_logit = alpha[i0] + alpha[i1] + torch.sum(Z[i0, :] * Z[i1, :], 1, keepdim=True)
proba_true = torch.sigmoid(A_logit)

Theta_X_true = (B0 + torch.matmul(Z, B))

true_values = {
    "ZZt": ZZt_true,
    "P": proba_true,
    "Theta_X": Theta_X_true,
    "Theta_A": A_logit,
    "BBt": torch.mm(B.t(), B),
    "alpha": alpha
}

init = {
    "weight": B,
    "positions": {"mean": Z}
}
# -----------------------------------------------------------------------------
# ADVI
# -----------------------------------------------------------------------------

# random initialization
advi = ADVI(K, N, p_cts, p_bin, mnar=mnar)

# advi.init(**init)

out, logs = advi.fit(train, train, eps=5.e-6, max_iter=1000, lr=0.1, return_log=True,
                     true_values=true_values)

logs.loc[:, ("error", slice(None))].plot()
plt.yscale("log")
plt.xscale("log")
plt.show()

# -----------------------------------------------------------------------------
# VIMC
# -----------------------------------------------------------------------------

vimc = VIMC(K, N, p_cts, p_bin, mnar=mnar, n_samples=50)
_, logs_vimc = vimc.fit(train, train, Z, batch_size=len(train),
         eps=5.e-6, max_iter=500, lr=0.1, alpha_true=alpha, return_log=True)

# -----------------------------------------------------------------------------
# MAP & MLE
# -----------------------------------------------------------------------------

mapfit = MAP(K, N, p_cts, p_bin, mnar=mnar)
_, logs_map = mapfit.fit(train, train, Z, batch_size=len(train),
         eps=5.e-6, max_iter=500, lr=0.1, alpha_true=alpha, return_log=True)


mlefit = MLE(K, N, p_cts, p_bin, mnar=mnar)
_, logs_mle = mlefit.fit(train, train, Z, batch_size=len(train),
         eps=5.e-6, max_iter=500, lr=0.1, alpha_true=alpha, return_log=True)



# -----------------------------------------------------------------------------
# NAIVI PLOT
# -----------------------------------------------------------------------------
ALGOS = {"ADVI": logs_advi, "VIMC": logs_vimc, "MLE": logs_mle, "MAP": logs_map}
fig, (ax_loss, ax_grad, ax_mse, ax_dist, ax_p) = plt.subplots(5, 1,
                        figsize=(8, 10), sharey="row", sharex="col"
                        )
for algo, logs in ALGOS.items():
    m = logs["llk_train"].min()
    ax_loss.plot((logs["llk_train"] - m) / m,
                 label=algo, color=COLORS[algo])
    ax_grad.plot(logs["grad_norm"]/ m,
                 label=algo, color=COLORS[algo])
    ax_mse.plot(logs["mse_train"],
                 label=algo, color=COLORS[algo])
    ax_dist.plot(logs["dist_inv"],
                 label=algo, color=COLORS[algo])
    ax_p.plot(logs["dist_proj"],
                 label=algo, color=COLORS[algo])

ax_loss.set_xscale("log")
ax_loss.set_yscale("log")
ax_grad.set_yscale("log")
ax_p.set_yscale("log")
ax_dist.set_yscale("log")
ax_loss.set_ylabel("Relative loss")
ax_grad.set_ylabel("Gradient norm (std.)")
ax_mse.set_ylabel("MSE")
ax_dist.set_ylabel("$D(\mathbf{Z}, \widehat{\mathbf{Z}})$")
ax_p.set_ylabel("$D(\mathbf{P}, \widehat{\mathbf{P}})$")
ax_dist.set_xlabel("Iteration")
# legend
lines = [Line2D([0], [0], color=COLORS[algo], linestyle="-")
         for algo in ALGOS]
labels = [DICT[algo]
                for algo in ALGOS]
fig.legend(lines, labels, loc=8, ncol=len(ALGOS))

fig.tight_layout()
fig.subplots_adjust(bottom=0.1)
fig.savefig("/home/simon/Documents/NAIVI/sims_mcmc/figs/convergence_naivi_smallp.pdf")

# -----------------------------------------------------------------------------
# MCMC
# -----------------------------------------------------------------------------

mcmc = MCMC(K, N, p_cts, p_bin, (0., 1.), (-2., 1.))
train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar, cuda=False)
mcmc.fit(train, max_iter=500, Z_true=Z.detach().cpu().numpy(), num_chains=5)

self = mcmc

ZZt_diag = az.summary(mcmc._fit, "ZZt", kind="diagnostics", round_to=3)
Z_diag = az.summary(mcmc._fit, "Z", kind="diagnostics", round_to=3)
B_diag = az.summary(mcmc._fit, "B", kind="diagnostics", round_to=3)
B0_diag = az.summary(mcmc._fit, "B0", kind="diagnostics", round_to=3)
alpha_diag = az.summary(mcmc._fit, "alpha", kind="diagnostics", round_to=3)
Theta_X_diag = az.summary(mcmc._fit, "Theta_X", kind="diagnostics", round_to=3)
Theta_A_diag = az.summary(mcmc._fit, "Theta_A", kind="diagnostics", round_to=3)



# plot
fig, axs = plt.subplots(7, 2, figsize=(8, 10), sharey="none", sharex="col")

bins = np.linspace(1., 2., 51)
ZZt_diag.hist(column="r_hat", ax=axs[0][0], bins=bins)
Z_diag.hist(column="r_hat", ax=axs[1][0], bins=bins)
alpha_diag.hist(column="r_hat", ax=axs[2][0], bins=bins)
Theta_X_diag.hist(column="r_hat", ax=axs[3][0], bins=bins)
Theta_A_diag.hist(column="r_hat", ax=axs[4][0], bins=bins)
B_diag.hist(column="r_hat", ax=axs[5][0], bins=bins)
B0_diag.hist(column="r_hat", ax=axs[6][0], bins=bins)

bins = np.linspace(0., 10000., 51)
ZZt_diag.hist(column="ess_tail", ax=axs[0][1], bins=bins)
Z_diag.hist(column="ess_tail", ax=axs[1][1], bins=bins)
alpha_diag.hist(column="ess_tail", ax=axs[2][1], bins=bins)
Theta_X_diag.hist(column="ess_tail", ax=axs[3][1], bins=bins)
Theta_A_diag.hist(column="ess_tail", ax=axs[4][1], bins=bins)
B_diag.hist(column="ess_tail", ax=axs[5][1], bins=bins)
B0_diag.hist(column="ess_tail", ax=axs[6][1], bins=bins)

for row in axs:
    for ax in row:
        ax.set_title("")

axs[0][0].set_title("$\widehat{R}$")
axs[0][1].set_title("ESS")
axs[0][0].set_ylabel("$\mathbf{Z}\mathbf{Z}^\\top$")
axs[1][0].set_ylabel("$\mathbf{Z}$")
axs[2][0].set_ylabel("$\\alpha$")
axs[3][0].set_ylabel("$\Theta^X$")
axs[4][0].set_ylabel("$\Theta^A$")
axs[5][0].set_ylabel("$\mathbf{B}$")
axs[6][0].set_ylabel("$\mathbf{B}_0$")

fig.tight_layout()
fig.savefig("/home/simon/Documents/NAIVI/sims_mcmc/figs/convergence_mcmc_smallp.pdf")