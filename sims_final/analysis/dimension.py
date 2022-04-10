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
from matplotlib.ticker import MaxNLocator

RESULTS_PATH = "/home/simon/Documents/NAIVI/sims_final/results/"
FIGS_PATH = "/home/simon/Documents/NAIVI/sims_final/figs/"
FIGSIZE = (10, 10)

FILE_NAME = "dimension.pdf"

FOLDER_NAME = "dimension"

EXPERIMENT_NAME = "missing_rate" # mistake

CURVES =[
	("fit", "algo"),
	("data", "K"),
	("model", "K")
]

METRICS = {
	"Train Loss": {
		"column": ("train", "loss"),
		"ytrans": None
	},
	"Train AUC": {
		"column": ("train", "auc"),
		"ytrans": None
	},
	"Test AUC": {
		"column": ("test", "auc"),
		"ytrans": None
	},
	"MSE(ZZt)": {
		"column": ("error", "ZZt"),
		"ytrans": "log"
	},
	"MSE(P)": {
		"column": ("error", "P"),
		"ytrans": "log"
	}
}

ALGOS = {
	"ADVI": {"color": "#ff0000", "display": "NAIVI-QB"},
	"MAP": {"color": "#00ff00", "display": "MAP"},
}

X_AXIS = ("model", "K")

# get data
df_list = []
for _, _, files in os.walk(RESULTS_PATH + FOLDER_NAME + "/"):
	for file in files:
		try:
			traj = Trajectory(EXPERIMENT_NAME, add_time=False)
			traj.f_load(filename=RESULTS_PATH + FOLDER_NAME + "/" + file, force=True)
			traj.v_auto_load = True
			df_list.append(pd.concat(
				[traj.res.summary.results.data, traj.res.summary.parameters.data],
				axis=1
			))
		except:
			pass
results = pd.concat(df_list)

# aggregate over curves
means = results.groupby(CURVES).agg("mean")
stds = results.groupby(CURVES).agg("std")
us = results.groupby(CURVES).agg("max")
ls = results.groupby(CURVES).agg("min")

# get columns
data_K = results[("data", "K")].unique()
algos = results[("fit", "algo")].unique()





# initiate plot
nrow = len(METRICS)
ncol = len(data_K)
FIGSIZE = (ncol * 3, nrow * 2)

fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE, sharex="col", sharey="row")
if nrow == 1:
	axs = [axs]
if ncol == 1:
	axs = [[ax] for ax in axs]

# cycle through columns
for col, K in enumerate(data_K):
	# plot
	for row, (metric, mparms) in enumerate(METRICS.items()):
		ax = axs[row][col]
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		for cname, curve in ALGOS.items():
			try:
				m = means.loc[(cname, K, slice(None)), mparms["column"]]
				x = means.loc[(cname, K, slice(None)),:].reset_index().loc[:, X_AXIS]
				s = stds.loc[(cname, K, slice(None)), mparms["column"]]
				i = ~m.isna()
				ax.plot(x[i.values], m.loc[i], color=curve["color"])
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


# legend
lines = [Line2D([0], [0], color=curve["color"], linestyle="-")
         for curve in ALGOS.values()]
labels = [curve["display"] for curve in ALGOS.values()]

# some parameters here ...
fig.legend(lines, labels, loc=8, ncol=3)
fig.tight_layout()
fig.subplots_adjust(bottom=0.15)

fig.savefig(FIGS_PATH + FILE_NAME)

