import pandas as pd
import numpy as np
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from pypet import Trajectory
import itertools as it
from matplotlib.lines import Line2D

RESULTS_PATH = "/home/simon/Documents/NAIVI/results/"
FIGS_PATH = "/home/simon/Documents/NAIVI/results/figs/"
EXP_NAME = "optimizer"

traj = Trajectory(EXP_NAME, add_time=False)
traj.f_load(filename=RESULTS_PATH + EXP_NAME + ".hdf5", force=True)
traj.v_auto_load = True

parms = traj.res.summary.parameters.df
ns = parms[("data", "N")].unique()
ps = parms[("data", "p_cts")].unique()
exps = list(it.product(ns, ps))
optimizers = parms[("fit", "optimizer")].unique()
cols = [matplotlib.colors.to_hex(col) for col in sns.color_palette("Set2", len(optimizers))]
colors = {
    name: col for name, col in zip(optimizers, cols)
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

# Plot
fig, axs = plt.subplots(len(metrics), len(exps), figsize=(15, 20), sharey="row", sharex="col")
for col, (N, p_cts) in enumerate(exps):
	runs = parms.loc[(parms[("data", "N")]==N) & (parms[("data", "p_cts")]==p_cts)].index
	min_loss = np.inf
	min_mse = np.inf
	max_auc = 0.
	for idx in runs:
		traj.f_set_crun(idx)
		min_loss = min(min_loss, traj.res.logs.crun.df[("train", "loss")].min())
		min_mse = min(min_mse, traj.res.logs.crun.df[("train", "mse")].min())
		max_auc = max(max_auc, traj.res.logs.crun.df[("train", "auc_A")].max())
	for row, (metric, display) in enumerate(metrics.items()):
		ax = axs[row][col]
		for idx in runs:
			traj.f_set_crun(idx)
			m = traj.res.logs.crun.df[metric]
			if metric == ("train", "loss"):
				m = (m - min_loss) / min_loss
			if metric == ("train", "mse"):
				m = (m - min_mse) / min_mse
			if metric == ("train", "auc_A"):
				m = max_auc - m
			ax.plot(m, color=colors[parms.loc[idx, ("fit", "optimizer")]])
		if col == 0:
			ax.set_yscale("log")
			ax.set_ylabel(display)
	axs[-1][col].set_xscale("log")
	axs[0][col].set_title(f"N={N}, p={p_cts}")
# legend
lines = [Line2D([0], [0], color=col, linestyle="-")
         for _, col in colors.items()]
labels = colors.keys()
fig.legend(lines, labels, loc=8, ncol=len(colors))
fig.tight_layout()
fig.subplots_adjust(bottom=0.05)
fig.savefig(FIGS_PATH + "optimizer.pdf")
