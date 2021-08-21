import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from NAIVI_experiments.display import colormap, to_display
plt.style.use("seaborn")

PATH = "./sims/"
EXPERIMENTS = [
    "covariate_binary",
    "covariate_continuous",
    "density_binary",
    "density_continuous",
    "missingrate_binary",
    "missingrate_continuous",
    "networksize_binary",
    "networksize_continuous"
]
BINARY = [True, False, ]*4
XAXIS = [
    "p_bin", "p_cts",
    "density", "density",
    "missing_rate", "missing_rate",
    "N", "N",
]
CURVES = [
    "N", "N",
    "p_bin", "p_cts",
    "p_bin", "p_cts",
    "p_bin", "p_cts",
]
ALGOS = ["VIMC", "ADVI", "MLE", "MICE"]
ALGOS = ["ADVI", "MLE", "MICE", "Mean", "NetworkSmoothing"]
colors = colormap
DICT = to_display
WHICH = [1000, 1000, 100, 100, 100, 100, 100, 100]


cov_type = "cts"
cov_type = "bin"

# fig, axs = plt.subplots(2, 4, sharex="col", figsize=(6, 3.5), sharey="row")
fig, axs = plt.subplots(1, 4, sharex="col", figsize=(6, 3.5), sharey="row")

for i, (exp, bin, xaxis, curves, which) in enumerate(zip(EXPERIMENTS, BINARY, XAXIS, CURVES, WHICH)):
    print(exp)
    if cov_type == "cts":
        j = i // 2
        if bin:
            continue
    else:
        j = (i+1) // 2
        if not bin:
            continue
    exp_short = exp.split("_")[0]
    yaxis = "test_auroc" if bin else "test_mse"
    file = PATH + exp + "/results/summary.csv"
    if os.path.isfile(file):
        results = pd.read_csv(file, index_col=0)
    else:
        print("---skipped (file not found)")
        continue
    algo = results["algo"]
    for a in ALGOS:
        df = results[algo == a]
        df = df[df[curves] == which]
        group = xaxis if xaxis != "density" else "alpha_mean"
        mean = df.groupby([group, curves]).agg("mean").reset_index()
        std = df.groupby([group, curves]).agg("std").reset_index()
        axs[j].plot(mean[xaxis], mean[yaxis], color=colors[a], label=DICT[a])
        axs[j].fill_between(mean[xaxis], mean[yaxis]-std[yaxis],
                               mean[yaxis]+std[yaxis], color=colors[a], alpha=0.2)
        # if a != "MICE":
        #     axs[1][j].plot(mean[xaxis], mean["dist_inv"], color=colors[a])
        #     axs[1][j].fill_between(mean[xaxis], mean["dist_inv"]-std["dist_inv"],
        #                            mean["dist_inv"]+std["dist_inv"], color=colors[a], alpha=0.2)
    if xaxis in ["N", "p_cts", "p_bin"]:
        axs[j].set_xscale("log")
    # axs[0][j].set_title("Setting {}".format("ABCDEFG"[j]))
    axs[j].set_xlabel(DICT[xaxis])
    # axs[1][j].set_ylim(0., 0.8)

# legend
lines = [Line2D([0], [0], color=colors[a]) for a in ALGOS]
labels = [DICT[a] for a in ALGOS]
fig.legend(lines, labels, loc=9, ncol=len(ALGOS)) #, title="Algorithm")
# ylabels
axs[0].set_ylabel("MSE" if cov_type == "cts" else "AUC")
# axs[1][0].set_ylabel("$D(\widehat Z, Z)$")
# axs[0].get_yaxis().set_label_coords(-0.25, 0.5)
# axs[1][0].get_yaxis().set_label_coords(-0.25, 0.5)
# layout
fig.tight_layout(h_pad=0.5, w_pad=0.)
fig.subplots_adjust(top=0.90)

plt.rcParams['savefig.facecolor'] = (1, 1, 1, 0)
for ax in axs:
    # for ax in axr:
    # ax.patch.set_facecolor('#C9C9C9')
    ax.patch.set_facecolor('#EEEEEE')
    for line in ax.get_xgridlines():
        # line.set_color("#CCCCCC")
        line.set_color("#FFFFFF")
        line.set_linewidth(0.5)
    for line in ax.get_ygridlines():
        # line.set_color("#CCCCCC")
        line.set_color("#FFFFFF")
        line.set_linewidth(0.5)
    for axis in ax.spines.values():
        axis.set_color("white")
        axis.set_linewidth(2)

# fig.savefig(PATH + "figs/{}_results_poster.pdf".format(cov_type))

fig.savefig(PATH + "figs/{}_results_slides.pdf".format(cov_type))