import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use("seaborn")

PATH = "/home/simon/Documents/NNVI/sims/"
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
ALGOS = ["MLE", "ADVI", "VIMC"]
colors = {"MLE": "#D16103", "ADVI": "#52854C", "VIMC": "#293352"}
WHICH = [1000, 1000, 100, 10, 100, 10, 100, 10]

# continuous

fig, axs = plt.subplots(2, 4, sharex="col", figsize=(10, 6), sharey="row")

for i, (exp, bin, xaxis, curves, which) in enumerate(zip(EXPERIMENTS, BINARY, XAXIS, CURVES, WHICH)):
    j = i // 2
    print(exp)
    if bin:
        continue
    exp_short = exp.split("_")[0]
    yaxis = "test_auroc" if bin else "test_mse"
    file = PATH + exp + "/results/summary.csv"
    if os.path.isfile(file):
        results = pd.read_csv(file, index_col=0)
    else:
        print("---skipped")
        continue
    algo = results["algo"]
    for a in ALGOS:
        df = results[algo==a]
        df = df[df[curves]==which]
        group = xaxis if xaxis != "density" else "alpha_mean"
        mean = df.groupby([group, curves]).agg("mean").reset_index()
        std = df.groupby([group, curves]).agg("std").reset_index()
        axs[0][j].plot(mean[xaxis], mean[yaxis], color=colors[a], label=a)
        axs[0][j].fill_between(mean[xaxis], mean[yaxis]-std[yaxis],
                               mean[yaxis]+std[yaxis], color=colors[a], alpha=0.2)

        axs[1][j].plot(mean[xaxis], mean["dist_inv"], color=colors[a])
        axs[1][j].fill_between(mean[xaxis], mean["dist_inv"]-std["dist_inv"],
                               mean["dist_inv"]+std["dist_inv"], color=colors[a], alpha=0.2)
    if xaxis in ["N", "p_cts", "p_bin"]:
        axs[0][j].set_xscale("log")
    axs[0][j].set_title("{}={}".format(curves, which))
    axs[1][j].set_xlabel(xaxis)
axs[0][0].legend(loc="upper left")
axs[0][0].set_ylabel("Test MSE")
axs[1][0].set_ylabel("Z Distance")
fig.tight_layout()
fig.savefig(PATH + "figs/binary_results.pdf")