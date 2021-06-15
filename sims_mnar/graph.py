import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from NAIVI_experiments.display import colormap, to_display
plt.style.use("seaborn")
PATH = "/home/simon/Documents/NAIVI/sims_mnar/"
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
groupings = ["algo", "mnar", "mnar_sparsity", "missing_mean", ]
means = results.groupby(groupings).agg("mean")
stds = results.groupby(groupings).agg("std")
us = results.groupby(groupings).agg("min")
ls = results.groupby(groupings).agg("max")

# sparsity
SPARSITIES = [0., 0.1, 0.5, 0.9, 1.]
METRICS = ["test_auroc", "dist_inv", "non_zero", "accuracy"]
ONLY_MNAR = [False, False, True, True]

# plot
fig, axs = plt.subplots(4, len(SPARSITIES),
                        figsize=(9, 8), sharex="all", sharey="row",
                        gridspec_kw={'height_ratios': [1.5, 1.5, 1, 1]}
                        )
for col, sparsity in enumerate(SPARSITIES):
    for algo, mnar in zip(ALGOS, MNARS):
        for row, (metric, only_mnar) in enumerate(zip(METRICS, ONLY_MNAR)):
            if only_mnar and not mnar:
                continue
            # auroc
            ax = axs[row][col]
            x = means.loc[(algo, mnar, sparsity, ), "missing_rate"]
            m = means.loc[(algo, mnar, sparsity, ), metric]
            i = ~m.isna()
            s = stds.loc[(algo, mnar, sparsity, ), metric]
            u = us.loc[(algo, mnar, sparsity, ), metric]
            l = ls.loc[(algo, mnar, sparsity, ), metric]
            ax.plot(x.loc[i], m.loc[i],
                        color=COLORS[algo],
                        label=DICT[algo] + ((", " + ("MNAR" if mnar else "MCAR"))
                        if algo in ["ADVI", "VIMC", "MLE"] else ""),
                        linestyle="--" if mnar else "-")
            # ax.fill_between(x.loc[i], m.loc[i]-s.loc[i], m.loc[i]+s.loc[i], color=COLORS[algo], alpha=0.2)
            # ax.fill_between(x.loc[i], l.loc[i], u.loc[i], color=COLORS[algo], alpha=0.2)
    axs[0][col].set_title("{}% sparse".format(int(sparsity*100)))
    axs[2][col].hlines(200 - sparsity*100, 0., 0.75, color="black", linestyle="-")
    axs[2][col].set_xticks([0., 0.25, 0.50, 0.75])
    axs[2][col].set_xticklabels([0, 25, 50, 75])
axs[-1][2].set_xlabel("Missing rate (%)")
axs[0][0].set_ylabel("AUC")
axs[1][0].set_ylabel("$D(\widehat Z, Z)$")
axs[2][0].set_ylabel("Non-zero")
axs[3][0].set_ylabel("Sel. acc.")
axs[0][0].set_xlim(0., 0.75)
axs[1][0].set_ylim(0., 0.08)
# legend
lines = [Line2D([0], [0], color=COLORS[algo], linestyle="--" if mnar else "-")
         for algo, mnar in zip(ALGOS, MNARS)] + [Line2D([0], [0], color="black", linestyle="-")]
labels = [DICT[algo] + ((", " + ("MNAR" if mnar else "MCAR"))
                if algo in ["ADVI", "VIMC", "MLE"] else "")
for algo, mnar in zip(ALGOS, MNARS)] + ["Oracle"]
fig.legend(lines, labels, loc=8, ncol=4) #, title="Algorithm")

fig.tight_layout()
fig.subplots_adjust(bottom=0.14)
fig.savefig(PATH + "figs/results.pdf")