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

# ===================================================
# settings
RESULTS_PATH = "/home/simon/Documents/NAIVI/results/"
FIGS_PATH = "/home/simon/Documents/NAIVI/results/figs/"
EXP_NAME = "network_weight"

curves = ("model", "network_weight")
curve_label = "Network weight"
xaxis = ("data", "adjacency_noise")
aggregate_over = ("data", "seed")
yaxes = {  # column: [display, logscale]
	("train", "mse"): ["Train. MSE(X)", False],
	("train", "auc_A"): ["Train. AUC(A)", False],
	("test", "mse"): ["Test. MSE(X)", False],
	("error", "ZZt"): ["MSE(ZZt)", False],
	("error", "Theta_X"): ["MSE(ThetaX)", False],
	("error", "Theta_A"): ["MSE(ThetaA)", False],
	("error", "P"): ["MSE(P)", False],
}

title = "N=200, p=100 (cts), 10 rep., 12% density, K=5, 27% missing"
xlab = "Adjacency logit noise variance"
xlog = True

# ===================================================
# get data
traj = Trajectory(EXP_NAME, add_time=False)
traj.f_load(filename=RESULTS_PATH + EXP_NAME + ".hdf5", force=True)
traj.v_auto_load = True
parms = traj.res.summary.parameters.df[[curves, xaxis, aggregate_over]]
results = traj.res.summary.results.df.join(parms)

# setup for plot
curve_names = parms[curves].unique()
xticks = parms[xaxis].unique()

cols = [matplotlib.colors.to_hex(col) for col in sns.color_palette("rocket", len(curve_names))]
colors = {name: col for name, col in zip(curve_names, cols)}

# compute mean and std
means = results.groupby([curves, xaxis]).agg("mean")
stds = results.groupby([curves, xaxis]).agg("std")


# ======================================================
# plot
fig, axs = plt.subplots(len(yaxes), 1, figsize=(8, 12), sharey="row", sharex="col")

# metrics
for ax, (yaxis, (display, log)) in zip(axs, yaxes.items()):
	m = means[yaxis]
	s = stds[yaxis]
	for curve in curve_names:
		y = m.loc[(curve, slice(None))]
		x = y.index.get_level_values(xaxis)
		ax.plot(x, y, color=colors[curve])
	ax.set_ylabel(display)
axs[0].set_title(title)
axs[-1].set_xlabel(xlab)
if xlog:
	axs[0].set_xscale("log")
# legend
lines = [Line2D([0], [0], color=col, linestyle="-") for _, col in colors.items()]
labels = colors.keys()
fig.legend(lines, labels, loc=8, ncol=len(colors), title=curve_label)
fig.tight_layout()
fig.subplots_adjust(bottom=0.10)
fig.savefig(FIGS_PATH + EXP_NAME + ".pdf")