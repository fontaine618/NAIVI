import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from NAIVI_experiments.display import colormap, to_display
import torch
plt.style.use("seaborn")
PATH = "/home/simon/Documents/NAIVI/sims_mnar_size/"
COLORS = colormap
DICT = to_display
ALGOS = [
    "ADVI", "ADVI",
     "VIMC", "VIMC",
     "MLE", "MLE",
     # "NetworkSmoothing", "MICE", "MissForest",
    # "Mean"
]
MNARS = [
    True, False,
     True, False,
     True, False,
     # False, False, False,
    # False
]

# retrieve results
dir = os.listdir(PATH)
folders = [x for x in dir if x.find(".") < 0]
exps = [x for x in folders if x.endswith("ar")]
results = pd.concat([
    pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
    for ex in exps
])
results["model"] = [
    (algo if algo not in ["ADVI", "VIMC", "MLE"] else algo+", "+("MNAR" if mnar else "MCAR"))
    for algo, mnar in zip(results["algo"], results["mnar"])
]
# patch dist_proj
results["dist_proj"] = results["dist_proj"].apply(lambda x: float(x[7:13]) if isinstance(x, str) else x)
# means +/- std
groupings = ["algo", "mnar", "N", "missing_mean", ]
means = results.groupby(groupings).agg("mean")
stds = results.groupby(groupings).agg("std")
us = results.groupby(groupings).agg("min")
ls = results.groupby(groupings).agg("max")

# sparsity
SIZES = [20, 50, 100, 200, 500]
METRICS = ["test_mse", "dist_inv", "dist_proj"]
ONLY_MNAR = [False, False, False, True, True]

# plot
fig, axs = plt.subplots(3, len(SIZES),
                        figsize=(9, 5), sharex="all", sharey="row",
                        gridspec_kw={'height_ratios': [1, 1, 1]}
                        )
for col, size in enumerate(SIZES):
    for algo, mnar in zip(ALGOS, MNARS):
        for row, (metric, only_mnar) in enumerate(zip(METRICS, ONLY_MNAR)):
            if only_mnar and not mnar:
                continue
            # auroc
            ax = axs[row][col]
            x = means.loc[(algo, mnar, size, ), "missing_rate"]
            m = means.loc[(algo, mnar, size, ), metric]
            i = ~m.isna()
            s = stds.loc[(algo, mnar, size, ), metric]
            u = us.loc[(algo, mnar, size, ), metric]
            l = ls.loc[(algo, mnar, size, ), metric]
            ax.plot(x.loc[i], m.loc[i],
                        color=COLORS[algo],
                        label=DICT[algo] + ((", " + ("MNAR" if mnar else "MCAR"))
                        if algo in ["ADVI", "VIMC", "MLE"] else ""),
                        linestyle="--" if mnar else "-")
            # ax.fill_between(x.loc[i], m.loc[i]-s.loc[i], m.loc[i]+s.loc[i], color=COLORS[algo], alpha=0.2)
            # ax.fill_between(x.loc[i], l.loc[i], u.loc[i], color=COLORS[algo], alpha=0.2)
    axs[0][col].set_title("N={}".format(int(size)))
    axs[-1][col].set_xticks([0., 0.25, 0.50, 0.75])
    axs[-1][col].set_xticklabels([0, 25, 50, 75])
axs[-1][2].set_xlabel("Missing rate (%)")
axs[0][0].set_ylabel("AUC")
axs[0][0].set_ylabel("MSE")
axs[1][0].set_ylabel("$D(\widehat Z, Z)$")
axs[2][0].set_ylabel("$D(\widehat P, P)$")
axs[0][0].set_xlim(0., 0.75)
axs[1][0].set_ylim(0., 0.70)
axs[2][0].set_ylim(0., 0.016)
# legend
lines = [Line2D([0], [0], color=COLORS[algo], linestyle="--" if mnar else "-")
         for algo, mnar in zip(ALGOS, MNARS)]
labels = [DICT[algo] + ((", " + ("MNAR" if mnar else "MCAR"))
                if algo in ["ADVI", "VIMC", "MLE"] else "")
                for algo, mnar in zip(ALGOS, MNARS)]
fig.legend(lines, labels, loc=8, ncol=3)

fig.tight_layout()
fig.subplots_adjust(bottom=0.20)
fig.savefig(PATH + "figs/results.pdf")
