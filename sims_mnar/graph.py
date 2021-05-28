import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from matplotlib.lines import Line2D
from NAIVI_experiments.display import colormap, to_display
plt.style.use("seaborn")
PATH = "./sims_mnar_old/"
COLORS = colormap
DICT = to_display
ALGOS = [
    "ADVI", "ADVI",
     # "VIMC", "VIMC",
     # "MLE", "MLE",
     "NetworkSmoothing", "MICE", "MissForest", "Mean"]
MNARS = [
    True, False,
     # True, False,
     # True, False,
     False, False, False, False]
# X_POS = {"VIMC": -0.15, "ADVI": -0.05, "MLE": 0.05, "MICE": 0.15}

# retrieve results
dir = os.listdir(PATH)
folders = [x for x in dir if x.find(".") < 0]
exps = [x for x in folders if x.startswith("mnar")]
results = pd.concat([
    pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
    for ex in exps
])
results["model"] = [
    (algo if algo not in ["ADVI", "VIMC", "MLE"] else algo+", "+("MNAR" if mnar else "MCAR"))
    for algo, mnar in zip(results["algo"], results["mnar"])
]
# means +/- std
means = results.groupby(["algo", "mnar", "missing_mean", ]).agg("mean")
stds = results.groupby(["algo", "mnar", "missing_mean", ]).agg("std")

# plot
fig, axs = plt.subplots(1, 1, figsize=(8, 5), sharex="col", sharey="row")
for algo, mnar in zip(ALGOS, MNARS):
    x = means.loc[(algo, mnar, ), "missing_rate"]
    m = means.loc[(algo, mnar, ), "test_auroc"]
    s = stds.loc[(algo, mnar, ), "test_auroc"]
    axs.plot(x, m,
                color=COLORS[algo],
                label=DICT[algo] + ((", " + ("MNAR" if mnar else "MCAR"))
                if algo in ["ADVI", "VIMC", "MLE"] else ""),
                linestyle="--" if mnar else "-")
    # axs[i].fill_between(XS, m-s, m+s, color=COLORS[algo], alpha=0.2)
    # x = XS + X_POS[algo]
    # lines = [[(xx, mm-ss), (xx, mm+ss)] for xx, mm, ss in zip(x, m, s)]
    # lc = mc.LineCollection(lines, colors=COLORS[algo], linewidths=1, label=DICT[algo])
    # axs[i].add_collection(lc)
    # axs[i].scatter(x, m, c=COLORS[algo], linewidth=1, edgecolor="white", s=20)
axs.set_title("")
axs.set_xlabel("Missing rate")
axs.set_ylabel("AUC")
# legend
lines = [Line2D([0], [0], color=COLORS[algo], linestyle="--" if mnar else "-")
         for algo, mnar in zip(ALGOS, MNARS)]
labels = [DICT[algo] + ((", " + ("MNAR" if mnar else "MCAR"))
                if algo in ["ADVI", "VIMC", "MLE"] else "")
for algo, mnar in zip(ALGOS, MNARS)]
fig.legend(lines, labels, loc=8, ncol=3) #, title="Algorithm")

fig.tight_layout()
fig.subplots_adjust(bottom=0.3)
fig.savefig(PATH + "figs/results.pdf")