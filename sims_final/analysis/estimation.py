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

RESULTS_PATH = "/home/simon/Documents/NAIVI/sims_final/results/"
FIGS_PATH = "/home/simon/Documents/NAIVI/sims_final/figs/"
FIGSIZE = (10, 4)

FILE_NAME = "estimation.pdf"

# curves
CURVES = {
	"ADVI": {"color": "#ff0000", "display": "NAIVI-QB"},
	"MAP": {"color": "#00ff00", "display": "MAP"},
	"VIMC": {"color": "#ff00ff", "display": "NAIVI-MC"},
	"MICE": {"color": "#00ff00", "display": "MICE"},
}

# rows
METRICS = {
	"MSE(Z)": {
		"column": ("error", "ZZt"),
		"ytrans": "log"
	},
	"MSE(P)": {
		"column": ("error", "P"),
		"ytrans": "log"
	}
}

# columns
EXPERIMENTS = {
	"estimation_N": {
		"groupby": ("data", "p_cts"),
		"groups": [0, 50],
		"xlab": "Network size ($N$)",
		"title": "Binary attributes",
		"xtrans": "log"
	},
}

# get all data
for name, exparms in EXPERIMENTS.items():
	# gather results
	df_list = []
	for _, _, files in os.walk(RESULTS_PATH + name + "/"):
		for file in files:
			try:
				traj = Trajectory(name, add_time=False)
				traj.f_load(filename=RESULTS_PATH + name + "/" + file, force=True)
				traj.v_auto_load = True
				df = pd.concat(
					[traj.res.summary.results.data, traj.res.summary.parameters.data],
					axis=1
				)
				df["experiment"] = name
				df_list.append(df)
			except:
				pass
results = pd.concat(df_list)

# initiate plot
nrow = len(METRICS)
ncol = sum([len(x["groups"]) for x in EXPERIMENTS.values()])
fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE, sharex="col", sharey="row")
if nrow == 1:
	axs = [axs]
if ncol == 1:
	axs = [[ax] for ax in axs]

# cycle through columns
for col, (name, exparms) in enumerate(EXPERIMENTS.items()):

	# group and aggregate
	groupings = [exparms["groupby"], ("fit", "algo")]
	res = results.loc[results["experiment"] == name]
	means = res.groupby(groupings).agg("mean")
	stds = res.groupby(groupings).agg("std")
	us = res.groupby(groupings).agg("max")
	ls = res.groupby(groupings).agg("min")
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

