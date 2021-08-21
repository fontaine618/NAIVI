import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import collections as mc
from NAIVI_experiments.display import colormap, to_display

# setup
plt.style.use("seaborn")
PATH = "./facebook/"
COLORS = colormap
DICT = to_display

# ALGOS = ["VIMC", "ADVI", "MLE", "MICE"]
ALGOS = ["ADVI", "MLE", "MICE", "Mean", "NetworkSmoothing"]
CENTERS = [698, 0, 1684]
CENTERS = [0]#, 1684]
XS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) #, 12, 15, 20])
MISSING_RATE = 0.5

# retrieve results
dir = os.listdir(PATH)
folders = [x for x in dir if x.find(".") < 0]
exps = [x for x in folders if x[:3] == "Kfb"]
# exps = ['Kfb_0_vimc', 'Kfb_0_advi', 'Kfb_0_mle', 'Kfb_0_mice', 'Kfb_1684_mle', 'Kfb_1684_advi']
results = pd.concat([
    pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
    for ex in exps
])
# patch simple
for alg in ["NetworkSmoothing", "MICE", "MissForest", "Mean"]:
    mice = results.loc[results["algo"] == alg]
    for k in np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]):
        micek = mice.copy()
        micek["K_model"] = k
        results = pd.concat([results, micek])

# means +/- std
# drop = results["train_loss"] == 0.
# results.drop(index=drop, inplace=True)
results["N"] = results["N"].astype(float)
means = results.groupby(["center", "algo", "K_model"]).agg("mean")
stds = results.groupby(["center", "algo", "K_model"]).agg("std")

# plot
nrow = 1
ncol = len(CENTERS)
# fig, axs = plt.subplots(nrow, ncol, figsize=(1.66*ncol, 2.5), sharey="row")
fig, axs = plt.subplots(nrow, ncol, figsize=(5.5, 3.5), sharey="row")
for i, c in enumerate(CENTERS):
    n = means.loc[(c, ), "N"].mean().astype(int)
    for algo in ALGOS:
        m = means.loc[(c, algo, ), "test_auroc"]
        s = stds.loc[(c, algo, ), "test_auroc"]
        xs = m.index
        axs.plot(xs, m, color=COLORS[algo], label=DICT[algo])
        axs.fill_between(xs, m-s, m+s, color=COLORS[algo], alpha=0.2)
    axs.set_xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    # axs[i].set_xticks(xs)
    # axs[i].set_xticklabels(xs)
    axs.set_title("#{} ($N=${})".format(c, n))
    axs.set_xlabel("Latent dimension")
    axs.set_xlim(1, 20)
axs.set_ylabel("AUC")
# axs[-1].legend(loc="lower right")
# legend
lines = [Line2D([0], [0], color=COLORS[a]) for a in ALGOS]
labels = [DICT[a] for a in ALGOS]
fig.legend(lines, labels, loc=8, ncol=len(ALGOS)) #, title="Algorithm")
fig.tight_layout(h_pad=0.5, w_pad=0.2)
# fig.subplots_adjust(bottom=0.35)
fig.subplots_adjust(bottom=0.25)
axs.patch.set_facecolor('#EEEEEE')
# for ax in axs:
#     ax.patch.set_facecolor('#EEEEEE')

# fig.savefig(PATH + "figs/Kfb_results.pdf")
fig.savefig(PATH + "figs/Kfb_results_slides.pdf")