import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from NAIVI_experiments.display import colormap, to_display

plt.style.use("seaborn")
PATH = "/home/simon/Documents/NAIVI/sims_mcmc/"
COLORS = colormap
DICT = to_display

# =============================================================================
# consistency networksize
ALGOS = [
	"ADVI",
	"VIMC",
	"MCMC",
]
# retrieve results
dir = os.listdir(PATH + "results/")
exps = [x for x in dir if x.find(".") < 0]
results = pd.concat([
    pd.read_csv("{}/results/{}/summary.csv".format(PATH, ex), index_col=0)
    for ex in exps
])


# means +/- std
groupings = ["p_cts", "algo", "N", ]
means = results.groupby(groupings).agg("mean")
stds = results.groupby(groupings).agg("std")
us = results.groupby(groupings).agg("min")
ls = results.groupby(groupings).agg("max")
METRICS = ["DZ", "DP", "DThetaX", "time"]


# plot
fig, axs = plt.subplots(len(METRICS), 2,
                        figsize=(8, 8), sharex="all", sharey="row",
                        gridspec_kw={'height_ratios': [1, 1, 1, 1]}
                        )
for col, p_cts in enumerate([0, 50]):
	for row, metric in enumerate(METRICS):
		if col == 0:
			axs[row][0].set_ylabel(DICT[metric])
		for algo in ALGOS:
			ax = axs[row][col]
			x = means.loc[(p_cts, algo, ), ].index
			m = means.loc[(p_cts, algo, ), metric]
			if m.max() < 1e-10:
				break
			i = ~m.isna()
			s = stds.loc[(p_cts, algo, ), metric]
			u = us.loc[(p_cts, algo, ), metric]
			l = ls.loc[(p_cts, algo, ), metric]
			ax.plot(x[i], m.loc[i],
			        color=COLORS[algo],
			        label=DICT[algo])
			ax.fill_between(x[i], l.loc[i], u.loc[i], color=COLORS[algo], alpha=0.2)

			ax.set_xscale("log")
			ax.set_yscale("log")
	axs[-1][col].set_xlabel("Network size ($N$)")
axs[0][0].set_title("No attributes")
axs[0][1].set_title("50 continuous attributes")

# legend
lines = [Line2D([0], [0], color=COLORS[algo], linestyle="-")
         for algo in ALGOS]
labels = [DICT[algo]
                for algo in ALGOS]
fig.legend(lines, labels, loc=8, ncol=3)

fig.tight_layout()
fig.subplots_adjust(bottom=0.15)
fig.savefig(PATH + "figs/consistency_networksize.pdf")
