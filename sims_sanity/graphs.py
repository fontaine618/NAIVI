import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from NAIVI_experiments.display import colormap, to_display
import matplotlib.ticker as ticker
import torch

plt.style.use("seaborn")
PATH = "/home/simon/Documents/NAIVI/sims_sanity/"
COLORS = colormap
DICT = to_display

# =============================================================================
# consistency networksize
ALGOS = [
	"ADVI",
	"VIMC",
	"MLE",
]
# ex = "r_consistency_networksize"
ex = "r_consistency_networksize_cov50"
ex = "r_consistency_networksize_cov200"
results = pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
# patch 2k
# ex = "r_consistency_networksize_2000"
# ex = "r_consistency_networksize_cov50_2000"
# results2k = pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
# results = pd.concat([results, results2k])
# means +/- std
groupings = ["algo", "N", ]
means = results.groupby(groupings).agg("mean")
stds = results.groupby(groupings).agg("std")
us = results.groupby(groupings).agg("min")
ls = results.groupby(groupings).agg("max")
METRICS = ["dist_inv", "dist_proj", ]
# plot
fig, axs = plt.subplots(1, len(METRICS),
                        figsize=(9, 4), sharex="all", sharey="none",
                        gridspec_kw={'height_ratios': [1, ]}
                        )
for col, metric in enumerate(METRICS):
	for algo in ALGOS:
		ax = axs[col]
		x = means.loc[(algo, ), ].index
		m = means.loc[(algo, ), metric]
		i = ~m.isna()
		s = stds.loc[(algo, ), metric]
		u = us.loc[(algo, ), metric]
		l = ls.loc[(algo, ), metric]
		ax.plot(x[i], m.loc[i],
		        color=COLORS[algo],
		        label=DICT[algo])
		ax.fill_between(x[i], m.loc[i]-s.loc[i], m.loc[i]+s.loc[i], color=COLORS[algo], alpha=0.2)
		ax.set_xlabel("Network size ($N$)")

axs[0].set_ylabel("$D(\widehat Z, Z)$")
axs[1].set_ylabel("$D(\widehat P, P)$")
axs[0].set_yscale("log")
axs[1].set_yscale("log")
axs[0].set_xscale("log")
axs[1].set_xscale("log")

# legend
lines = [Line2D([0], [0], color=COLORS[algo], linestyle="-")
         for algo in ALGOS]
labels = [DICT[algo]
                for algo in ALGOS]
fig.legend(lines, labels, loc=8, ncol=3)

fig.tight_layout()
fig.subplots_adjust(bottom=0.20)
# fig.savefig(PATH + "figs/consistency_networksize.pdf")
# fig.savefig(PATH + "figs/consistency_networksize_cov50.pdf")
fig.savefig(PATH + "figs/consistency_networksize_cov200.pdf")




# =============================================================================
# consistency nbattributes
ALGOS = [
	"ADVI",
	"VIMC",
	"MLE",
]
ex = "r_consistency_nbattributes"
results = pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
# patch VIMC5
results = results.loc[results["algo"] != "VIMC"]
ex = "r_consistency_nbattributes_VIMC5"
results_vimc = pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
results = pd.concat([results, results_vimc])
# means +/- std
groupings = ["algo", "p_cts", ]
means = results.groupby(groupings).agg("mean")
stds = results.groupby(groupings).agg("std")
us = results.groupby(groupings).agg("min")
ls = results.groupby(groupings).agg("max")
METRICS = ["dist_inv", "dist_proj", ] #"train_mse", ]
# plot
fig, axs = plt.subplots(1, len(METRICS),
                        figsize=(9, 4), sharex="all", sharey="none",
                        gridspec_kw={'height_ratios': [1, ]}
                        )
for col, metric in enumerate(METRICS):
	for algo in ALGOS:
		ax = axs[col]
		x = means.loc[(algo, ), ].index
		m = means.loc[(algo, ), metric]
		i = ~m.isna()
		s = stds.loc[(algo, ), metric]
		u = us.loc[(algo, ), metric]
		l = ls.loc[(algo, ), metric]
		ax.plot(x[i]+1, m.loc[i],
		        color=COLORS[algo],
		        label=DICT[algo])
		ax.fill_between(x[i]+1, m.loc[i]-s.loc[i], m.loc[i]+s.loc[i], color=COLORS[algo], alpha=0.2)
		ax.set_xlabel("Nb. attributes ($p$)")
		ax.set_xscale("log")
		ax.set_yscale("log")
		ax.set_xticks([1, 10, 100, 1000])
		ax.set_xticklabels([0, 10, 100, 1000])

axs[0].set_ylabel("$D(\widehat Z, Z)$")
axs[1].set_ylabel("$D(\widehat P, P)$")
# axs[2].set_ylabel("Train. MSE")
# axs[2].set_ylim(0.8, 1.3)

# legend
lines = [Line2D([0], [0], color=COLORS[algo], linestyle="-")
         for algo in ALGOS]
labels = [DICT[algo]
                for algo in ALGOS]
fig.legend(lines, labels, loc=8, ncol=3)

fig.tight_layout()
fig.subplots_adjust(bottom=0.20)
fig.savefig(PATH + "figs/consistency_nbattributes.pdf")






# =============================================================================
# vimc_nsamples
ALGOS = [1, 2, 5, 10, 20, 50]
ex = "r_vimc_nsamples"
results = pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
ex = "r_vimc_nsamples_large"
results_large = pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
results = pd.concat([results, results_large])
# means +/- std
groupings = ["n_sample", "N", ]
means = results.groupby(groupings).agg("mean")
stds = results.groupby(groupings).agg("std")
us = results.groupby(groupings).agg("min")
ls = results.groupby(groupings).agg("max")
METRICS = ["dist_inv", "dist_proj", "train_mse", ]
COLORS = {1: '#000000', 2: '#222222', 5: '#444444',
          10: '#666666', 20: '#888888', 50: '#aaaaaa'}
# plot
fig, axs = plt.subplots(1, len(METRICS),
                        figsize=(9, 4), sharex="all", sharey="none",
                        gridspec_kw={'height_ratios': [1, ]}
                        )
for col, metric in enumerate(METRICS):
	for algo in ALGOS:
		ax = axs[col]
		x = means.loc[(algo, ), ].index
		m = means.loc[(algo, ), metric]
		i = ~m.isna()
		s = stds.loc[(algo, ), metric]
		u = us.loc[(algo, ), metric]
		l = ls.loc[(algo, ), metric]
		ax.plot(x[i], m.loc[i],
		        color=COLORS[algo],
		        label=algo)
		# ax.fill_between(x[i], m.loc[i]-s.loc[i], m.loc[i]+s.loc[i], color=COLORS[algo], alpha=0.2)
		ax.set_xlabel("Network size ($N$)")
		ax.set_xscale("log")

axs[0].set_ylabel("$D(\widehat Z, Z)$")
axs[1].set_ylabel("$D(\widehat P, P)$")
axs[2].set_ylabel("Train. MSE")
# axs[0].set_yscale("log")

# legend
lines = [Line2D([0], [0], color=COLORS[algo], linestyle="-")
         for algo in ALGOS]
labels = [algo for algo in ALGOS]
fig.legend(lines, labels, loc=8, ncol=len(ALGOS), title="MC Samples")

fig.tight_layout()
fig.subplots_adjust(bottom=0.30)
fig.savefig(PATH + "figs/vimc_nsamples.pdf")