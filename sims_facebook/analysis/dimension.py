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
import matplotlib
from facebook.data import get_data

PATH = "/home/simon/Documents/NAIVI/facebook/data/raw/"
RESULTS_PATH = "/home/simon/Documents/NAIVI/sims_facebook/results/"
FIGS_PATH = "/home/simon/Documents/NAIVI/sims_facebook/figs/"
FIGSIZE = (10, 10)

FILE_NAME = "dimension.pdf"

FOLDER_NAME = "fb_dimension"

EXPERIMENT_NAME = "fb_dimension"

CURVES = [
	("fit", "algo"),
	("data", "center"),
	("model", "K")
]

METRICS = {
	"Train Loss (/min)": {
		"column": ("train", "loss"),
		"ytrans": None
	},
	"Train AUC(A)": {
		"column": ("train", "auc_A"),
		"ytrans": None
	},
	"Train AUC(X)": {
		"column": ("train", "auc"),
		"ytrans": None
	},
	"Test AUC(X)": {
		"column": ("test", "auc"),
		"ytrans": None
	},
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

# patch repeated indices
cols = results.columns.values
cols[13] = ("data", "missing_rate")
results.columns = pd.MultiIndex.from_tuples(cols)


# aggregate over curves
means = results.groupby(CURVES).agg("mean")
stds = results.groupby(CURVES).agg("std")
us = results.groupby(CURVES).agg("max")
ls = results.groupby(CURVES).agg("min")

# get columns
centers = results[("data", "center")].unique()
algos = results[("fit", "algo")].unique()


centers = {
	c: matplotlib.colors.to_hex(col)
    for c, col in zip(centers, sns.color_palette("tab10", len(centers)))
}
# get N, p
Ns = {}
for c in centers.keys():
	_, _, _, _, X_bin = get_data(PATH, c)
	Ns[c] = tuple(X_bin.shape)

# initiate plot
nrow = len(METRICS)
ncol = len(algos)
FIGSIZE = (ncol * 3 + 1, nrow * 2)


plt.cla()
fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE, sharex="col", sharey="row")
if nrow == 1:
	axs = [axs]
if ncol == 1:
	axs = [[ax] for ax in axs]

# cycle through columns
for col, algo in enumerate(algos):
	# plot
	for row, (metric, mparms) in enumerate(METRICS.items()):
		ax = axs[row][col]
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		for c, color in centers.items():
			m = means.loc[(algo, c, slice(None)), mparms["column"]]
			if mparms["column"] == ("train", "loss"):
				m = m / m.min()
			x = means.loc[(algo, c, slice(None)), :].reset_index().loc[:, X_AXIS]
			s = stds.loc[(algo, c, slice(None)), mparms["column"]]
			i = ~m.isna()
			ax.plot(x[i.values], m.loc[i], color=color)

	axs[0][col].set_title("Algorithm: " + ALGOS[algo]["display"])
	axs[-1][col].set_xlabel("Fitted dimension")

# row things
for row, (metric, mparms) in enumerate(METRICS.items()):
	axs[row][0].set_ylabel(metric)
	if mparms["ytrans"] is not None:
		axs[row][0].set_yscale(mparms["ytrans"])


# legend
lines = [Line2D([0], [0], color=color, linestyle="-")
         for c, color in centers.items()]
labels = [f"{c} {Ns[c]}" for c in centers.keys()]

# some parameters here ...
fig.legend(lines, labels, title="Center (N, p)", loc=5, ncol=1)
fig.tight_layout()
fig.subplots_adjust(right=0.60)

fig.savefig(FIGS_PATH + FILE_NAME)

