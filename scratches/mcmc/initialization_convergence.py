from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI, VIMC, MLE, MICE, MCMC, MAP, GLM
from NAIVI.initialization import initialize
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
N = 200
K = 5
p_cts = 20
p_bin = 0
var_cts = 1.
missing_mean = -2.
alpha_mean = -2.
seed = 4
mnar_sparsity = 1.0
mnar = False
constant_components = False

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0, W = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_mean=missing_mean,
    alpha_mean=alpha_mean, seed=seed, mnar_sparsity=mnar_sparsity, constant_components=constant_components
)

E = i0.shape[0]
p = p_bin + p_cts

train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar, test=True)

ZZt_true = Z @ Z.T
A_logit = alpha + alpha.t() + ZZt_true
proba_true = torch.sigmoid(A_logit)

Theta_X_true = (B0 + torch.matmul(Z, B))

true_values = {
    "ZZt": ZZt_true,
    "P": proba_true[i0, i1].unsqueeze(1),
    "Theta_X": Theta_X_true,
    "Theta_A": A_logit[i0, i1].unsqueeze(1),
    "BBt": torch.mm(B.t(), B),
    "alpha": alpha
}
# -----------------------------------------------------------------------------
# Various initializations
# -----------------------------------------------------------------------------
# glm = GLM(K, N, p_cts, p_bin, mnar=False, latent_positions=Z)
# glm.fit(train, None, eps=1.e-6, max_iter=200, lr=1.)
# outt = initialize(train, K)
#
# random = {
# }
BZ = {
    "weight": B,
    "positions": {"mean": Z}
}
# Bt = {
#     "weight": B
# }
# Zt = {
#     "positions": {"mean": Z}
# }
# Zglm = {
#     "positions": {"mean": Z},
#     "weight": glm.model.mean_model.weight.data.t()
# }
# usvt_glm = outt
# BZalpha = {
#     "weight": B,
#     "positions": {"mean": Z},
#     "heterogeneity": {"mean": alpha}
# }

inits = {
    # "B random, Z random": random,
    "B true, Z true": BZ,
    # "B true, Z true, alpha true": BZalpha,
    # "B true, Z random": Bt,
    # "B random, Z true": Zt,
    # "B glm, Z true": Zglm,
    # "B glm, Z usvt": usvt_glm
}
# -----------------------------------------------------------------------------
# ADVI
# -----------------------------------------------------------------------------
results = dict()

for name, init in inits.items():
    results[name] = dict()

    fit = ADVI(K, N, p_cts, p_bin, mnar=mnar, estimate_components=True, n_samples=10)
    fit.init(**init)
    out, logs = fit.fit(train, test, eps=1.e-8, max_iter=200, lr=0.001, return_log=True,
                     true_values=true_values)

    results[name]["out"] = out
    results[name]["logs"] = logs




fit.model.adjacency_model.components
W


# -----------------------------------------------------------------------------
# NAIVI PLOT
# -----------------------------------------------------------------------------
import matplotlib
import seaborn as sns
cols = [matplotlib.colors.to_hex(col) for col in sns.color_palette("Set2", len(results))]
colors = {
    name: col for name, col in zip(results.keys(), cols)
}
metrics = {
    ("train", "loss"): "(Training loss - min)/min",
    ("train", "grad_Linfty"): "Linfty(grad)",
    ("train", "grad_L1"): "L1(grad)",
    ("train", "grad_L2"): "L2(grad)",
    ("train", "mse"): "(Training MSE - min)/min",
    ("train", "auc_A"): "(max - Training AUC)",
    ("error", "BBt"): "MSE(BBt)",
    ("error", "Theta_X"): "MSE(Theta_X)",
    ("error", "ZZt"): "MSE(ZZt)",
    ("error", "alpha"): "MSE(alpha)",
    ("error", "Theta_A"): "MSE(Theta_A)",
    ("error", "P"): "MSE(P)",
}
min_loss = min(
    min(res["logs"][("train", "loss")]) for name, res in results.items()
)

min_mse = min(
    min(res["logs"][("train", "mse")]) for name, res in results.items()
)
max_auc = max(
    max(res["logs"][("train", "auc_A")]) for name, res in results.items()
)

fig, axs = plt.subplots(len(metrics), 1, figsize=(8, 20), sharey="row", sharex="col")
for ax, (metric, display) in zip(axs, metrics.items()):
    for name, col in colors.items():
        m = results[name]["logs"][metric]
        if metric == ("train", "loss"):
            m = (m - min_loss) / min_loss
        if metric == ("train", "mse"):
            m = (m - min_mse) / min_mse
        if metric == ("train", "auc_A"):
            m = max_auc - m
        ax.plot(m, color=col)
    ax.set_yscale("log")
    ax.set_ylabel(display)
axs[0].set_xscale("log")


# legend
lines = [Line2D([0], [0], color=col, linestyle="-")
         for _, col in colors.items()]
labels = colors.keys()
fig.legend(lines, labels, loc=8, ncol=len(colors)//2)

fig.tight_layout()
fig.subplots_adjust(bottom=0.05)
fig.savefig("/home/simon/Documents/NAIVI/sims_mcmc/figs/convergence_advi_n200p100.pdf")
