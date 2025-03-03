import torch
from pypet import cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
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

    # "MAP":              ("MAP",         "#3333ff", "dotted", "s"),
    # "MLE":              ("MLE",         "#3333ff", "dotted", "v"),
    # "NetworkSmoothing": ("Smooth",      "#6633ff", "dashed", "s"),

    # "FA":               ("GLFM",        "#99cc66", "dotted", "o"),
    # "KNN":              ("KNN",         "#88ff88", "dashed", "v"),
    # "MICE":             ("MICE",        "#88ff88", "dashed", "s"),

    # "Mean":             ("Mean",        "#55cc55", "dotted", "s"),
}

# missing mechanisms
missing_mechanisms = {
    # "uniform": "Uniform",
    "row_deletion": "Row deletion",
    "triangle": "Triangle",
    # "block": "Block",
}



name = "facebook"
res_list = []
for i in range(30):
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    # file = f"./experiments/{name}/results/old/seed{i}_K.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)


    res_list.append(results)

results = pd.concat(res_list)
metric = "testing.auroc_binary_weighted_average"
# metric = "testing.auroc_binary"
centers = { #center :(N, p)
    3980: (59, 42),
    698: (66, 48),
    414: (159, 103),
    686: (170, 62),
    348: (227, 126),
    0: (347, 139),
    3437: (547, 116),
    1912: (755, 133),
    1684: (792, 100),
    107: (1045, 153)
}

n_methods = len(methods)

fig, axs = plt.subplots(4, 1, figsize=(10, 8),
                        sharex="col", sharey="row", squeeze=True,
                        gridspec_kw={"height_ratios": [1, 0.4] * 2})

for row, (missing_mechanism, missing_mechanism_name) in enumerate(missing_mechanisms.items()):
    res = results.loc[
        (results["method"].isin(methods.keys()))
        & (results["data.missing_mechanism"] == missing_mechanism)
    ]

    # Metric
    ax = axs[2*row]

    for x, (center, (N, p)) in enumerate(centers.items()):
        res_center = res.loc[res["data.facebook_center"] == center]

        for i, (method, (method_name, color, linestyle, marker)) in enumerate(methods.items()):
            res_method = res_center.loc[res_center["method"] == method]
            med = res_method[metric].median()
            lower = res_method[metric].quantile(0.25)
            upper = res_method[metric].quantile(0.75)
            xi = x + (i - n_methods/2 + 0.5) / (n_methods+5)
            ax.scatter(xi, med, color=color, marker=marker, facecolor='none')
            ax.plot([xi, xi], [0., med], color=color, linestyle=linestyle)
    # ax2 = ax.twinx()
    # ax2.set_ylabel(missing_mechanism_name, rotation=270, labelpad=15)
    # ax2.set_yticks([])
    # ax2.set_yticklabels([])

    ax.set_xticks(np.linspace(0, 9, 10))
    ax.set_xticklabels([f"{center}\n({N}, {p})" for center, (N, p) in centers.items()])
    ax.grid(axis="x")
    ax.set_ylabel("Pred. AuROC")
    ax.set_ylim(0.4, 0.75)
    ax.set_title(f"{missing_mechanism_name}")
    # Wilcoxon signed rank test
    ax = axs[2*row+1]
    ax.axhline(y=np.log10(0.05), color="black", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax.axhline(y=-np.log10(0.05), color="black", linestyle="--", alpha=0.5)


    for x, (center, (N, p)) in enumerate(centers.items()):
        res_center = res.loc[res["data.facebook_center"] == center]
        res_vmp = res_center.loc[res_center["method"] == "VMP"]
        y_vmp = res_vmp[metric].values.astype(float)

        for i, (method, (method_name, color, linestyle, marker)) in enumerate(methods.items()):
            if method == "VMP":
                continue
            res_other = res_center.loc[res_center["method"] == method]
            y_other = res_other[metric].values.astype(float)
            stat, p = wilcoxon(y_vmp, y_other, nan_policy="omit", alternative="two-sided")
            stat_l, p_l = wilcoxon(y_vmp, y_other, nan_policy="omit", alternative="less")
            stat_g, p_g = wilcoxon(y_vmp, y_other, nan_policy="omit", alternative="greater")
            sign = (p_g > p_l)*1.
            y = sign * - np.log10(p_l) + (1. - sign) * np.log10(p_g)
            xi = x + (i - n_methods/2 + 0.5) / (n_methods+5)
            ax.plot(xi, y,
                    label=methods[method][0], color=methods[method][1],
                    linestyle="none", marker=methods[method][3],
                    markerfacecolor='none')
    ax.set_ylabel("Signed -log $p$-value \n NAIVI vs. Other")


axs[3].set_xlabel("Ego network center (N, p)")
# legend
lines = [Line2D([0], [0], color=color, linestyle=ltype, marker=mtype, markerfacecolor='none')
         for _, (_, color, ltype, mtype) in methods.items()]
labels = [name for _, (name, _, _, _) in methods.items()]
# fig.legend(lines, labels, loc=9, ncol=5)
fig.legend(lines, labels, loc=9, ncol=2)
plt.tight_layout()
fig.subplots_adjust(top=0.90)
# plt.savefig(f"./experiments/{name}/facebook_metrics.pdf")
plt.savefig(f"./experiments/{name}/facebook_metrics0.pdf")



