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
	# "VIMC",
	# "MCMC",
	"MLE",  "MAP"
]


# retrieve results for networksize
dir = os.listdir(PATH + "results/")
exps = [x for x in dir if x.find(".") < 0]
results = pd.concat([
    pd.read_csv("{}results/{}/summary.csv".format(PATH, ex), index_col=0)
    for ex in exps if (ex.find("p") < 0 and "summary.csv" in os.listdir(PATH + "results/" + ex))
])

results_p = pd.concat([
    pd.read_csv("{}results/{}/summary.csv".format(PATH, ex), index_col=0)
    for ex in exps if (ex.find("p") >= 0 and "summary.csv" in os.listdir(PATH + "results/" + ex))
])

results_p.sort_values(["algo", "N", "p_cts", "seed"])[["algo", "N", "p_cts", "seed"]]
# means +/- std
groupings = ["p_cts", "algo", "N", ]
means = results.groupby(groupings).agg("mean")
stds = results.groupby(groupings).agg("std")
us = results.groupby(groupings).agg("max")
ls = results.groupby(groupings).agg("min")


groupings = ["N", "algo", "p_cts", ]
means_p = results_p.groupby(groupings).agg("mean")
stds_p = results_p.groupby(groupings).agg("std")
us_p = results_p.groupby(groupings).agg("max")
ls_p = results_p.groupby(groupings).agg("min")
METRICS = ["DZ", "DP", "DThetaX", "time"]


# plot
fig, axs = plt.subplots(len(METRICS), 3,
                        figsize=(8, 8), sharey="row", sharex="col",
                        gridspec_kw={'height_ratios': [1, 1, 1, 1]}
                        )
for col, p_cts in enumerate([0, 50, "p"]):
	for row, metric in enumerate(METRICS):
		if col == 0:
			axs[row][0].set_ylabel(DICT[metric])
		for algo in ALGOS:
			ax = axs[row][col]
			if p_cts == "p":
				x = means_p.loc[(200, algo, ), ].index
				# to correct the bad scaling
				mult = 200 / x if metric == "DThetaX" else 1.
				m = means_p.loc[(200, algo, ), metric] * mult
				if m.max() < 1e-10:
					break
				i = (~m.isna()) & (m.gt(0.))
				u = us_p.loc[(200, algo, ), metric] * mult
				l = ls_p.loc[(200, algo, ), metric] * mult
				ax.plot(x[i]+1, m.loc[i],
				        color=COLORS[algo],
				        label=DICT[algo])
				ax.fill_between(x[i]+1, l.loc[i], u.loc[i], color=COLORS[algo], alpha=0.2)
			else:
				x = means.loc[(p_cts, algo, ), ].index
				m = means.loc[(p_cts, algo, ), metric]
				if m.max() < 1e-10:
					break
				i = ~m.isna()
				u = us.loc[(p_cts, algo, ), metric]
				l = ls.loc[(p_cts, algo, ), metric]
				ax.plot(x[i], m.loc[i],
				        color=COLORS[algo],
				        label=DICT[algo])
				ax.fill_between(x[i], l.loc[i], u.loc[i], color=COLORS[algo], alpha=0.2)

			ax.set_xscale("log")
			ax.set_yscale("log")
axs[-1][0].set_xlabel("Network size ($N$)")
axs[-1][1].set_xlabel("Network size ($N$)")
axs[-1][2].set_xlabel("Nb. attributes ($p$)")
axs[0][0].set_title("No attributes")
axs[0][1].set_title("50 continuous attributes")
axs[0][2].set_title("$N=200$ nodes")

axs[-1][2].set_xticks([1, 11, 101, 1001])
axs[-1][2].set_xticklabels([0, 10, 100, 1000])

# legend
lines = [Line2D([0], [0], color=COLORS[algo], linestyle="-")
         for algo in ALGOS]
labels = [DICT[algo]
                for algo in ALGOS]
fig.legend(lines, labels, loc=8, ncol=len(ALGOS))

fig.tight_layout()
fig.subplots_adjust(bottom=0.15)
fig.savefig(PATH + "figs/consistency_networksize_new.pdf")
