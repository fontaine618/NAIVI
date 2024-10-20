import torch
from pypet import cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import math
from scipy.stats import wilcoxon

torch.set_default_tensor_type(torch.cuda.FloatTensor)
plt.rcParams.update(plt.rcParamsDefault)
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.default": "regular",
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Lato"],
    "axes.labelweight": "normal",
    "figure.titleweight": "bold",
    "figure.titlesize": "large",
    "font.weight": "normal",
    # "axes.formatter.use_mathtext": True,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# Methods
methods = {
    # "Oracle":           ("Oracle",      "#000000", "solid", "s"),

    "VMP0":             ("NAIVI-0",     "#9966ff", "dotted", "v"),
    "VMP":              ("NAIVI",       "#3366ff", "solid", "o"),
    # "MCMC":             ("MCMC",       "#3366ff", "dotted", "s"),

    "MAP":              ("MAP",         "#3333ff", "dotted", "s"),
    "MLE":              ("MLE",         "#3333ff", "dotted", "v"),
    "NetworkSmoothing": ("Smooth",      "#6633ff", "dashed", "s"),

    "FA":               ("GLFM",        "#99cc66", "dotted", "o"),
    "KNN":              ("KNN",         "#88ff88", "dashed", "v"),
    "MICE":             ("MICE",        "#88ff88", "dashed", "s"),

    "Mean":             ("Mean",        "#55cc55", "dotted", "s"),
}
seeds = range(30)

name = "email"
res_list = []
for i in seeds:
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    res_list.append(results)

email = pd.concat(res_list)

name = "cora"
res_list = []
for i in seeds:
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    results = results.loc[results["method"] != "GCN"]
    res_list.append(results)

cora = pd.concat(res_list)

email["dataset"] = "email"
cora["dataset"] = "cora"
results = pd.concat([email, cora])


metrics = {
    "testing.f1_multiclass_weighted": ("F1 (weighted)", ),
    # "testing.accuracy_multiclass": ("Accuracy", ),
}


fig, axs = plt.subplots(nrows=len(metrics)*2, ncols=2, figsize=(10, 3*len(metrics)+2),
                        sharey="row", sharex="col", squeeze=False,
                        gridspec_kw={"height_ratios": [1, 0.6] * len(metrics)})
for col, (dataset, dataset_name) in enumerate([("email", "Email"), ("cora", "Cora")]):
    res_dataset = results.loc[results["dataset"] == dataset]
    for row, (metric, (metric_name, )) in enumerate(metrics.items()):
        # metric
        ax = axs[2*row, col]
        for i, (method, (method_name, color, linestyle, marker)) in enumerate(methods.items()):
            res_method = res_dataset.loc[res_dataset["method"] == method]
            ys = res_method.groupby("data.n_seeds").agg(
                median=(metric, "median"),
                lower=(metric, lambda x: np.quantile(x, 0.25)),
                upper=(metric, lambda x: np.quantile(x, 0.75)),
            )
            xs = ys.index

            ax.plot(xs, ys["median"], color=color, linestyle=linestyle,
                    marker=marker, label=method_name, markerfacecolor='none')
            # ax.fill_between(xs, ys["lower"], ys["upper"], color=color, alpha=0.2)
        ax.set_xticks([2, 4, 6, 8, 10])
        if col == 0:
            ax.set_ylabel(metric_name, size=10)
        # wilcoxon
        ax = axs[2*row+1, col]
        ax.axhline(y=np.log10(0.05), color="black", linestyle="--", alpha=0.5)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.axhline(y=-np.log10(0.05), color="black", linestyle="--", alpha=0.5)
        if col == 0:
            ax.set_ylabel("Signed -log $p$-value \n NAIVI vs. Other")
        if row == len(metrics) - 1:
            ax.set_xlabel("Seeds / class", size=10)

        for i, (method, (method_name, color, linestyle, marker)) in enumerate(methods.items()):
            if method == "VMP":
                continue
            res_other = res_dataset.loc[res_dataset["method"] == method]
            res_vmp = res_dataset.loc[res_dataset["method"] == "VMP"]
            signed_pvals = []
            for x in xs:
                other = res_other.loc[res_other["data.n_seeds"] == x][metric].values.astype(float)
                vmp = res_vmp.loc[res_vmp["data.n_seeds"] == x][metric].values.astype(float)
                stat, p = wilcoxon(vmp, other, nan_policy="omit", alternative="two-sided")
                stat_l, p_l = wilcoxon(vmp, other, nan_policy="omit", alternative="less")
                stat_g, p_g = wilcoxon(vmp, other, nan_policy="omit", alternative="greater")
                s = (p_g > p_l)*1.
                y = s * -math.log(p_l) + (1-s) * math.log(p_g)
                signed_pvals.append(y)
            ax.plot(xs, np.array(signed_pvals), color=color, linestyle="none", marker=marker, markerfacecolor='none')


    axs[0, col].set_title(dataset_name, size=12)
# legend
lines = [Line2D([0], [0], color=color, linestyle=ltype, marker=mtype, markerfacecolor='none')
         for _, (_, color, ltype, mtype) in methods.items()]
labels = [name for _, (name, _, _, _) in methods.items()]
fig.legend(lines, labels, loc=9, ncol=9)
plt.tight_layout()
fig.subplots_adjust(top=0.85)
# plt.show()

plt.savefig(f"./experiments/semi_supervised_metrics.pdf")