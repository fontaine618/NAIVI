import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.lines import Line2D
from NAIVI_experiments.display import colormap, to_display
plt.style.use("seaborn")
PATH = "./facebook/"
COLORS = colormap
DICT = to_display
# ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']
MISSING_RATES = [0.25, 0.5]
ALGOS = ["ADVI", "VIMC", "MLE", "NetworkSmoothing", "MICE", "MissForest", "Mean"]
X_POS = {"VIMC": -0.15, "ADVI": -0.05, "MLE": 0.05, "MICE": 0.15}
CENTERS = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
XS = np.arange(len(CENTERS))
XAXIS_ORDER = "N"
na = len(ALGOS)
bar_width = 0.6 / na

# retrieve results
dir = os.listdir(PATH)
folders = [x for x in dir if x.find(".") < 0]
exps = [x for x in folders if x[:2] == "fb"]
results = pd.concat([
    pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
    for ex in exps
])
# means +/- std
means = results.groupby(["missing_rate", "algo", "center"]).agg("mean")
stds = results.groupby(["missing_rate", "algo", "center"]).agg("std")

# get order
tmp = results.groupby("center").agg("mean")[XAXIS_ORDER].sort_values()
centers = tmp.index.values

# plot
nrow = len(MISSING_RATES)
ncol = 1
# fig, axs = plt.subplots(nrow, ncol, figsize=(2.5*ncol, 2.5), sharex="col", sharey="row")
fig, axs = plt.subplots(nrow, ncol, figsize=(8, 6), sharex="col", sharey="row")
for i, rate in enumerate(MISSING_RATES):
    for j, algo in enumerate(ALGOS):
        m = means.loc[(rate, algo, ), "test_auroc"].loc[centers]
        s = stds.loc[(rate, algo, ), "test_auroc"].loc[centers]
        # axs[i].plot(XS, m, color=COLORS[algo], label=DICT[algo])
        axs[i].bar(x=XS + (j-na/2)*bar_width,
                   height=2.*s,
                   bottom=m-s,
                   color=COLORS[algo], label=DICT[algo],
                   width=bar_width, alpha=0.5)
        axs[i].plot(XS + (j-na/2)*bar_width, m, 's', ms=4.,
                   color=COLORS[algo], label=DICT[algo])
        # axs[i].fill_between(XS, m-s, m+s, color=COLORS[algo], alpha=0.2)
        # x = XS + X_POS[algo]
        # lines = [[(xx, mm-ss), (xx, mm+ss)] for xx, mm, ss in zip(x, m, s)]
        # lc = mc.LineCollection(lines, colors=COLORS[algo], linewidths=1, label=DICT[algo])
        # axs[i].add_collection(lc)
        # axs[i].scatter(x, m, c=COLORS[algo], linewidth=1, edgecolor="white", s=20)
    axs[i].set_xticks(XS)
    axs[i].set_xticklabels(tmp.astype(int), rotation="vertical")
    axs[i].set_title("{:.0f} % missing".format(rate*100))
    axs[i].set_ylabel("AUC")
axs[1].set_xlabel("Network size")
axs[0].set_ylim(0.48, 0.95)
# legend
lines = [Line2D([0], [0], color=COLORS[a]) for a in ALGOS]
labels = [DICT[a] for a in ALGOS]
fig.legend(lines, labels, loc=8, ncol=len(ALGOS)) #, title="Algorithm")

fig.tight_layout()
# fig.subplots_adjust(bottom=0.40)
fig.subplots_adjust(bottom=0.2)
fig.savefig(PATH + "figs/fb_results_bar.pdf")