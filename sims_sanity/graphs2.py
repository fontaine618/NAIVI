import pandas as pd
import numpy as np
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
from pypet import Trajectory
import itertools as it
from matplotlib.lines import Line2D

RESULTS_PATH = "/home/simon/Documents/NAIVI/sims_sanity/results/"
FIGS_PATH = "/home/simon/Documents/NAIVI/sims_sanity/figs/"
FIGSIZE = (10, 6)

FILE_NAME = "estimation.pdf"

# curves
CURVE_COLUMN = ("fit", "algo")
CURVE_TITLE = "Algorithm"
CURVES = {
	"ADVI": {"color": "#ff0000", "display": "NAIVI-QB"},
	"MAP": {"color": "#00ff00", "display": "MAP"},
	"MLE": {"color": "#00ffff", "display": "MLE"},
	"VIMC": {"color": "#ff00ff", "display": "NAIVI-MC"},
	"MICE": {"color": "#00ff00", "display": "MICE"},
	# "MissForest": {"color": "#009900", "display": "MissForest"},
	# "NetworkSmoothing": {"color": "#0000ff", "display": "NetworkSmoothing"},
	# "Mean": {"color": "#000099", "display": "Mean"}
}

# rows
METRICS = {
	"MSE($\mathbf{Z}$)": {
		"column": ("error", "ZZt"),
		"ytrans": "log"
	},
	"MSE($\mathbf{P}$)": {
		"column": ("error", "P"),
		"ytrans": "log"
	},
	"MSE($\mathbf{B}\mathbe{B}^\\top$)": {
		"column": ("error", "P"),
		"ytrans": "log"
	}
}

# columns
EXPERIMENTS = {
	"missing_N": {
		"groupby": ("data", "N"),
		"display": ("data", "N"),
		"xlab": "Network size ($N$)",
		"xtrans": "log"
	},
	"missing_density": {
		"groupby": ("data", "alpha_mean"),
		"display": ("data", "density"),
		"xlab": "Network density",
		"xtrans": "log"
	},
	"missing_rate": {
		"groupby": ("data", "missing_mean"),
		"display": ("data", "missing_prop"),
		"xlab": "Missing proportion",
		"xtrans": None
	},
	"missing_p": {
		"groupby": ("data", "p_bin"),
		"display": ("data", "p_bin"),
		"xlab": "Nb. attributes ($p$)",
		"xtrans": "log"
	},
}


# initiate plot
nrow = len(METRICS)
ncol = len(EXPERIMENTS)
fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE, sharex="col", sharey="row")
if nrow == 1:
	axs = [axs]
if ncol == 1:
	axs = [[ax] for ax in axs]

# cycle through columns
for col, (name, exparms) in enumerate(EXPERIMENTS.items()):
	# gather results
	df_list = []
	for _, _, files in os.walk(RESULTS_PATH + name + "/"):
		for file in files:
			try:
				traj = Trajectory(name, add_time=False)
				traj.f_load(filename=RESULTS_PATH + name + "/" + file, force=True)
				traj.v_auto_load = True
				df_list.append(pd.concat(
					[traj.res.summary.results.data, traj.res.summary.hyperparameters.data],
					axis=1
				))
			except:
				pass
	results = pd.concat(df_list)
	# group and aggregate
	groupings = [CURVE_COLUMN, exparms["groupby"]]
	means = results.groupby(groupings).agg("mean")
	stds = results.groupby(groupings).agg("std")
	us = results.groupby(groupings).agg("max")
	ls = results.groupby(groupings).agg("min")
	# plot
	for row, (metric, mparms) in enumerate(METRICS.items()):
		ax = axs[row][col]
		for cname, curve in CURVES.items():
			try:
				m = means.loc[(cname, slice(None)), mparms["column"]]
				x = means.loc[(cname, slice(None)),:].reset_index().loc[:, exparms["display"]]
				s = stds.loc[(cname, slice(None)), mparms["column"]]
				i = ~m.isna()
				ax.plot(x[i.values], m.loc[i], color=curve["color"])
			except:
				pass

	axs[0][col].set_title("Experiment " + "ABCDEFGH"[col])
	axs[-1][col].set_xlabel(exparms["xlab"])
	if exparms["xtrans"] is not None:
		axs[0][col].set_xscale(exparms["xtrans"])

# row things
for row, (metric, mparms) in enumerate(METRICS.items()):
	axs[row][0].set_ylabel(metric)
	if mparms["ytrans"] is not None:
		axs[row][0].set_yscale(mparms["ytrans"])

# column things
for col, (name, exparms) in enumerate(EXPERIMENTS.items()):
	pass

axs[0][1].set_xticks([0.02, 0.1, 0.5, ])
axs[0][1].set_xticklabels([0.02, 0.1, 0.5, ])

# legend
lines = [Line2D([0], [0], color=curve["color"], linestyle="-")
         for curve in CURVES.values()]
labels = [curve["display"] for curve in CURVES.values()]

# some parameters here ...
fig.legend(lines, labels, loc=8, ncol=6)
fig.tight_layout()
fig.subplots_adjust(bottom=0.20)

fig.savefig(FIGS_PATH + FILE_NAME)

