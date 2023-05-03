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
    "Oracle":           ("Oracle",      "#000000", "solid", "s"),

    "VMP":              ("NAIVI",       "#3366ff", "solid", "o"),
    # "ADVI":             ("NAIVI-QB",    "#8888ff", "dashed", "v"),

    "MAP":              ("MAP",         "#3333ff", "dotted", "s"),
    "MLE":              ("MLE",         "#3333ff", "dotted", "v"),
    "NetworkSmoothing": ("Smooth",      "#6633ff", "dashed", "s"),

    "FA":               ("GLFM",        "#99cc66", "dotted", "o"),
    "KNN":              ("KNN",         "#88ff88", "dashed", "v"),
    "MICE":             ("MICE",        "#88ff88", "dashed", "s"),

    "Mean":             ("Mean",        "#55cc55", "dotted", "s"),
}


# missing mechanisms
missing_mechanisms = {
    "uniform": "Uniform",
    "row_deletion": "Row deletion",
    # "triangle": "Triangle",
    # "block": "Block",
}

torch.set_default_tensor_type(torch.cuda.FloatTensor)
name = "dim_attributes"
res_list = []
for i in range(31):
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    res_list.append(results)

results = pd.concat(res_list)

# plt.cla()
# sns.lineplot(
#     data=results.loc[results["data.missing_mechanism"] == "uniform"],
#     x="data.latent_dim_attributes",
#     y="testing.auroc_binary",
#     hue="method",
# )
# plt.show()



fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for i, (missing_mechanism, missing_mechanism_name) in enumerate(missing_mechanisms.items()):
    res = results.loc[
        (results["method"].isin(methods.keys()))
        & (results["data.missing_mechanism"] == missing_mechanism)
    ]

    ax = axs[i]

    for i, (method, (method_name, color, linestyle, marker)) in enumerate(methods.items()):
        res_method = res.loc[res["method"] == method]
        ys = res_method.groupby("data.latent_dim_attributes").agg(
            median=("testing.auroc_binary", "median"),
        )
        ax.plot(10 - ys.index, ys["median"], color=color, linestyle=linestyle,
                marker=marker, label=method_name, markerfacecolor='none')

    ax.set_title(f"{missing_mechanism_name}")
    ax.set_xlabel("Edge latent dimension")
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xticklabels([0, 2, 4, 6, 8, 10])
axs[0].set_ylabel("Pred. AuROC")
# legend
lines = [Line2D([0], [0], color=color, linestyle=ltype, marker=mtype, markerfacecolor='none')
         for _, (_, color, ltype, mtype) in methods.items()]
labels = [name for _, (name, _, _, _) in methods.items()]
fig.legend(lines, labels, loc=9, ncol=9)
plt.tight_layout()
fig.subplots_adjust(top=0.82)
plt.savefig(f"./experiments/{name}/dim_attr_metrics.pdf")
