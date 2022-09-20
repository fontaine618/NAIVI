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

FILE_NAME = "dimension.pdf"

FOLDER_NAME = "fb_dimension"

EXPERIMENT_NAME = ["fb_dimension", "fb_dimension_vimc"]

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
	"Test AUC": {
		"column": ("test", "auc"),
		"ytrans": None
	},
}

ALGOS = {
	"ADVI": {"linestyle": "-", "display": "NAIVI-QB"},
	"VIMC": {"linestyle": ":", "display": "NAIVI-MC"},
	# "MAP": {"linestyle": "--", "display": "MAP"},
}

X_AXIS = ("model", "K")

# get data
df_list = []
for exp in EXPERIMENT_NAME:
	for _, _, files in os.walk(RESULTS_PATH + exp + "/"):
		for file in files:
			try:
				traj = Trajectory(exp, add_time=False)
				traj.f_load(filename=RESULTS_PATH + exp + "/" + file, force=True)
				traj.v_auto_load = True
				df = pd.concat(
					[traj.res.summary.results.data, traj.res.summary.parameters.data],
					axis=1
				)
				print(exp, df.columns[13], df.columns[15])

				# patch repeated indices
				cols = df.columns.values
				cols[15] = ("data", "missing_rate")
				df.columns = pd.MultiIndex.from_tuples(cols)
				print(exp, df.columns[13], df.columns[15])
				df_list.append(df)
			except:
				pass
results = pd.concat(df_list)


# aggregate over curves
means = results.groupby(CURVES).agg("mean")
stds = results.groupby(CURVES).agg("std")
us = results.groupby(CURVES).agg("max")
ls = results.groupby(CURVES).agg("min")

# get columns
centers = results[("data", "center")].unique()
centers = {
	c: matplotlib.colors.to_hex(col)
    for c, col in zip(centers, sns.color_palette("tab10", len(centers)))
}
# get N, p
Ns = {}
for c in centers.keys():
	_, _, _, _, X_bin = get_data(PATH, c)
	Ns[c] = tuple(X_bin.shape)

# columns
COLUMNS = {
	"$N<200$": [3980, 698, 686, 414],
	"$200<N<700$": [0, 348, 3437],
	"$N>700$": [107, 1684, 1912],
}

# best K
best = {
	algo: {c: means.loc[(algo, c, slice(None)), ("train", "loss")].idxmin()[2]
	for c in centers.keys()} for algo in ALGOS
}

# initiate plot
nrow = len(METRICS)
ncol = len(COLUMNS)
FIGSIZE = (ncol * 3 + 1, nrow * 2)

plt.cla()
fig, axs = plt.subplots(nrow, ncol, figsize=FIGSIZE, sharex="col", sharey="row")
if nrow == 1:
	axs = [axs]
if ncol == 1:
	axs = [[ax] for ax in axs]

# cycle through columns
for col, (cname, cs) in enumerate(COLUMNS.items()):
	# plot
	for row, (metric, mparms) in enumerate(METRICS.items()):
		ax = axs[row][col]
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		for c in cs:
			for algo, aparms in ALGOS.items():
				color = centers[c]
				m = means.loc[(algo, c, slice(None)), mparms["column"]]
				if mparms["column"] == ("train", "loss"):
					m = m / m.min()
				x = means.loc[(algo, c, slice(None)), :].reset_index().loc[:, X_AXIS]
				i = ~m.isna()
				ax.plot(x[i.values], m.loc[i], color=color, linestyle=aparms["linestyle"])
				ax.scatter(best[algo][c], m.loc[(algo, c, best[algo][c])], color=color)
	axs[0][col].set_title(cname)
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
fig.subplots_adjust(right=0.80)

fig.savefig(FIGS_PATH + FILE_NAME)

