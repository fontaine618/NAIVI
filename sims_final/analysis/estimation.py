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
	"ADVI": {"linestyle": "-", "display": "NAIVI-QB"},
	"VIMC": {"linestyle": ":", "display": "NAIVI-MC"},
	# "MAP": {"linestyle": "--", "display": "MAP"},
	# "MCMC": {"linestyle": "-.", "display": "MICE"},
}

# rows
METRICS = {
	"$D(\mathbf{Z}, \widehat{\mathbf{Z}})$": {
		"column": ("error", "ZZt"),
		"ytrans": "log"
	},
	"$D(\mathbf{B}, \widehat{\mathbf{B}})$": {
		"column": ("error", "BBt"),
		"ytrans": "log"
	},
	"$D(\mathbf{P}, \widehat{\mathbf{P}})$": {
		"column": ("error", "P"),
		"ytrans": "log"
	}
}

# columns
EXPERIMENTS = {
	"estimation_N": {
		"groupby": ("data", "p_cts"),
		"groups": [0, 50],
		"xgroup": ("data", "N"),
		"xlab": "Network size ($N$)",
		"title": "$p=$",
		"xtrans": "log"
	},
	"estimation_p": {
		"groupby": ("data", "N"),
		"groups": [100, 500],
		"xgroup": ("data", "p_cts"),
		"xlab": "Nb attributes ($p$)",
		"title": "$N=$",
		"xtrans": "log"
	},
}

# get all data
df_list = []
for name, exparms in EXPERIMENTS.items():
	# gather results
	for _, _, files in os.walk(RESULTS_PATH + name + "/"):
		for file in files:
			try:
				traj = Trajectory(name, add_time=False)
				traj.f_load(filename=RESULTS_PATH + name + "/" + file, force=True)
				traj.v_auto_load = True
				print(name, file)
				df = pd.concat(
					[traj.res.summary.results.data, traj.res.summary.hyperparameters.data],
					axis=1
				)
				# ensure compatible columns (no idea why, but some are not in the MultiIndex format ...)
				df.columns = pd.MultiIndex.from_tuples(df.columns.values)
				df["experiment"] = name
				df_list.append(df)
			except:
				pass
results = pd.concat(df_list, ignore_index=True, axis=0)


# initiate plot
nrow = len(METRICS)
ncol = sum([len(x["groups"]) for x in EXPERIMENTS.values()])
FIGSIZE = (ncol * 3, nrow * 2)
fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE, sharex="col", sharey="row")
if nrow == 1:
	axs = [axs]
if ncol == 1:
	axs = [[ax] for ax in axs]

col = -1
# cycle through columns
for name, exparms in EXPERIMENTS.items():
	# group and aggregate
	groupings = [exparms["groupby"], ("fit", "algo"), exparms["xgroup"]]
	res = results.loc[results["experiment"] == name]
	means = res.groupby(groupings).agg("mean")
	stds = res.groupby(groupings).agg("std")
	us = res.groupby(groupings).agg("max")
	ls = res.groupby(groupings).agg("min")
	for val in exparms["groups"]:
		col = col+1
		# plot
		for row, (metric, mparms) in enumerate(METRICS.items()):
			ax = axs[row][col]
			for cname, curve in CURVES.items():
				try:
					m = means.loc[(val, cname, slice(None)), mparms["column"]]
					l = us.loc[(val, cname, slice(None)), mparms["column"]]
					u = ls.loc[(val, cname, slice(None)), mparms["column"]]
					x = means.loc[(val, cname, slice(None)),:].reset_index().loc[:, exparms["xgroup"]]
					i = ~m.isna()
					ax.plot(x[i.values], m.loc[i], linestyle=curve["linestyle"], color="black")
					ax.fill_between(x[i.values], l[i.values], u[i.values], color="black", alpha=0.2)
				except:
					pass

		axs[0][col].set_title(f"{exparms['title']}{val}")
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

# axs[0][1].set_xticks([0.02, 0.1, 0.5, ])
# axs[0][1].set_xticklabels([0.02, 0.1, 0.5, ])
axs[1][0].set_ylim(10**-3, 10**-0)

# legend
lines = [Line2D([0], [0], linestyle=curve["linestyle"], color="black")
         for curve in CURVES.values()]
labels = [curve["display"] for curve in CURVES.values()]

# some parameters here ...
fig.legend(lines, labels, loc=8, ncol=6)
fig.tight_layout()
fig.subplots_adjust(bottom=0.20)

fig.savefig(FIGS_PATH + FILE_NAME)

