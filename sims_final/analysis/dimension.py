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
from matplotlib.ticker import MaxNLocator

RESULTS_PATH = "/home/simon/Documents/NAIVI/sims_final/results/"
FIGS_PATH = "/home/simon/Documents/NAIVI/sims_final/figs/"
FIGSIZE = (10, 10)
FILE_NAME = "dimension.pdf"

# columns are true K
# rows are metrics
# curves are N

EXPERIMENTS = {
	"dimension": {
		"name": "missing_rate" # mistake
	},
}

CURVES =[
	("data", "K"),
	("data", "N"),
	("fit", "algo"),
	("model", "K")
]

METRICS = {
	"Train Loss (/min)": {
		"column": ("train", "loss"),
		"ytrans": None
	},
	# "Train AUC": {
	# 	"column": ("train", "auc"),
	# 	"ytrans": None
	# },
	"Test AUC": {
		"column": ("test", "auc"),
		"ytrans": None
	},
	"$D(\mathbf{Z}, \widehat{\mathbf{Z}})$": {
		"column": ("error", "ZZt"),
		"ytrans": "log"
	},
	# "MSE(P)": {
	# 	"column": ("error", "P"),
	# 	"ytrans": "log"
	# }
}

ALGOS = {
	"ADVI": {"linestyle": "-", "display": "NAIVI-QB"},
	"VIMC": {"linestyle": ":", "display": "NAIVI-MC"},
	"MAP": {"linestyle": "--", "display": "MAP"},
}

NS = {
	50: {"color": "#0000ff", "display": "$N=50$"},
	200: {"color": "#00ff00", "display": "$N=50$"},
	500: {"color": "#ff0000", "display": "$N=500$"},
}

X_AXIS = ("model", "K")

# get all results
df_list = []
for name, exparms in EXPERIMENTS.items():
	for _, _, files in os.walk(RESULTS_PATH + name + "/"):
		for file in files:
			print(file)
			try:
				traj = Trajectory(exparms["name"], add_time=False)
				traj.f_load(filename=RESULTS_PATH + name + "/" + file, force=True)
				traj.v_auto_load = True
				df_list.append(pd.concat(
					[traj.res.summary.results.data, traj.res.summary.parameters.data],
					axis=1
				))
			except:
				pass
results = pd.concat(df_list)
results = results.loc[results[("fit", "algo")].apply(lambda x: x in ALGOS), :]

# aggregate over curves
means = results.groupby(CURVES).agg("mean")
stds = results.groupby(CURVES).agg("std")
us = results.groupby(CURVES).agg("max")
ls = results.groupby(CURVES).agg("min")

# get columns
data_K = sorted(results[("data", "K")].unique())

# best K
best = means.groupby(CURVES[0:3]).apply(lambda df: df.loc[:, ("train", "loss")].idxmin()[3])

# initiate plot
nrow = len(METRICS)
ncol = len(data_K)
FIGSIZE = (ncol * 3, nrow * 3)

plt.cla()
fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE, sharex="col", sharey="row")
if nrow == 1:
	axs = [axs]
if ncol == 1:
	axs = [[ax] for ax in axs]

# cycle through columns
for col, K in enumerate(data_K):
	# cycle through metrics
	for row, (metric, mparms) in enumerate(METRICS.items()):
		ax = axs[row][col]
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		# algorithms
		for algo, curve in ALGOS.items():
			# Ns
			for n, nparms in NS.items():
				try:
					m = means.loc[(K, n, algo, slice(None)), mparms["column"]]
					s = stds.loc[(K, n, algo, slice(None)), mparms["column"]]
					if mparms["column"] == ("train", "loss"):
						m = m / m.min()
					# 	s = s / n
					x = means.loc[(K, n, algo, slice(None)),:].reset_index().loc[:, X_AXIS]
					i = ~m.isna()
					ax.plot(x[i.values], m.loc[i], color=nparms["color"], linestyle=curve["linestyle"])
					# ax.fill_between(x[i.values], m.loc[i] - s.loc[i], m.loc[i] + s.loc[i],
					#                 alpha=0.2, color=nparms["color"])
					i = best[(K, n, algo)]
					ax.scatter(i, m.loc[(K, n, algo, i)], color=nparms["color"])
				except:
					pass
		ax.axvline(K, linestyle="--", color="black")

	axs[0][col].set_title("True dimension: " + str(K))
	axs[-1][col].set_xlabel("Fitted dimension")

# row things
for row, (metric, mparms) in enumerate(METRICS.items()):
	axs[row][0].set_ylabel(metric)
	if mparms["ytrans"] is not None:
		axs[row][0].set_yscale(mparms["ytrans"])

axs[0][0].set_ylim(0.99, 1.25)

# # legend
lines = [Line2D([0], [0], color=nparms["color"], linestyle=curve["linestyle"])
         for nparms in NS.values() for curve in ALGOS.values()]
labels = [f'{curve["display"]} ($N={n}$)' for n in NS.keys() for curve in ALGOS.values()]

# some parameters here ...
fig.legend(lines, labels, loc=8, ncol=3)
fig.tight_layout()
fig.subplots_adjust(bottom=0.15)

fig.savefig(FIGS_PATH + FILE_NAME)

