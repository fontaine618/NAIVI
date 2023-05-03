import torch
from pypet import cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D


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

    "VMP":              ("NAIVI",       "#3366ff", "solid", "o"),
    # "ADVI":             ("NAIVI-QB",    "#8888ff", "dashed", "v"),

    # "MAP":              ("MAP",         "#3333ff", "dotted", "s"),
    # "MLE":              ("MLE",         "#3333ff", "dotted", "v"),
    "NetworkSmoothing": ("Smooth",      "#6633ff", "dashed", "s"),
    # "GCN":              ("GCN",         "#8833ff", "dashed", "v"),

    "FA":               ("GLFM",        "#99cc66", "dotted", "o"),
    "KNN":              ("KNN",         "#88ff88", "dashed", "v"),
    "MICE":             ("MICE",        "#88ff88", "dashed", "s"),

    "Mean":             ("Mean",        "#55cc55", "dotted", "s"),
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
for i in range(31):
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)

    # there was a typo in the original job where 698 was replaced by 398,
    # so we need to patch things up a bit
    to_drop = results["data.facebook_center"] == 398
    results = results.loc[~to_drop, :]

    file = f"./experiments/{name}/results/seed{i}_698.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters698 = gather_parameters_to_DataFrame(traj)
    results698 = gather_results_to_DataFrame(traj)
    results698 = parameters698.join(results698)

    results = pd.concat([results, results698])


    res_list.append(results)

results = pd.concat(res_list)

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

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharey=True, sharex=True)

for i, (missing_mechanism, missing_mechanism_name) in enumerate(missing_mechanisms.items()):
    res = results.loc[
        (results["method"].isin(methods.keys()))
        & (results["data.missing_mechanism"] == missing_mechanism)
    ]

    ax = axs[i]

    for x, (center, (N, p)) in enumerate(centers.items()):
        res_center = res.loc[res["data.facebook_center"] == center]

        for i, (method, (method_name, color, linestyle, marker)) in enumerate(methods.items()):
            res_method = res_center.loc[res_center["method"] == method]
            med = res_method["testing.auroc_binary"].median()
            lower = res_method["testing.auroc_binary"].quantile(0.25)
            upper = res_method["testing.auroc_binary"].quantile(0.75)
            xi = x + (i - n_methods/2 + 0.5) / (n_methods+5)
            ax.scatter(xi, med, color=color, marker=marker, facecolor='none')
            ax.plot([xi, xi], [0., med], color=color, linestyle=linestyle)
            # ax.plot([xi, xi], [lower, upper], color=color, linestyle=linestyle)

    # if i == len(missing_mechanisms) - 1:
    #     ax.set_ylabel(missing_mechanism_name)
    #     ax.yaxis.set_label_position("right")
    # ax.set_title(f"{missing_mechanism_name}")
    # ax.figure.text(0.01, 0.5, missing_mechanism_name, va='center', rotation='vertical')
    ax2 = ax.twinx()
    ax2.set_ylabel(missing_mechanism_name, rotation=270, labelpad=15)
    ax2.set_yticks([])
    ax2.set_yticklabels([])

    ax.set_xticks(np.linspace(0, 9, 10))
    ax.set_xticklabels([f"{center}\n({N}, {p})" for center, (N, p) in centers.items()])
    ax.grid(axis="x")
    ax.set_ylabel("Pred. AuROC")
    ax.set_ylim(0.7, 0.9)

axs[1].set_xlabel("Ego network center (N, p)")
# legend
lines = [Line2D([0], [0], color=color, linestyle=ltype, marker=mtype, markerfacecolor='none')
         for _, (_, color, ltype, mtype) in methods.items()]
labels = [name for _, (name, _, _, _) in methods.items()]
fig.legend(lines, labels, loc=9, ncol=6)
plt.tight_layout()
fig.subplots_adjust(top=0.92)
plt.savefig(f"./experiments/{name}/facebook_metrics2.pdf")



