import torch
from pypet import cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
from matplotlib.lines import Line2D
from itertools import product
import math

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
    "triangle": "Triangle",
    "block": "Block",
}

experiments = {
    # "experiment_name": ("group_by", "display_var", "display_name", logx?)
    "n_nodes_binary": ("data.n_nodes", "data.n_nodes", "Nb. nodes", True),
    "n_attributes_binary": ("data.p_bin", "data.p_bin", "Nb. attributes", True),
    # "edge_density": ("data.heterogeneity_mean", "training.edge_density", "Edge density", False),
    # "missing_rate": ("data.missing_covariate_rate", "training.X_missing_prop", "Missing rate", False),
}
seeds = range(30)

# Parameters
rows_by = "data.missing_mechanism"
curves_by = "method"
cols_by = "experiment"

# performance metric
metric = "training.cpu_time"
yaxis = "CPU Time (s)"

full_df_list = []

for name, (group_by, display_var, display_name, _) in experiments.items():

    res_list = []
    for i in seeds:
        file = f"./experiments/{name}/results/seed{i}.hdf5"
        tname = name + "_seed" + str(i)
        traj = Trajectory(name=tname)
        traj.f_load(filename=file, load_results=2, force=True)

        parameters = gather_parameters_to_DataFrame(traj)
        results = gather_results_to_DataFrame(traj)
        results = parameters.join(results)
        res_list.append(results)

    results = pd.concat(res_list)
    results["experiment"] = name


    cols = list(set([rows_by, curves_by, display_var, cols_by, group_by, metric, "data.seed"]))
    df = results.loc[:, cols]
    df["x_value"] = df.groupby([group_by, rows_by])[display_var].transform("median")

    outdf = df.groupby([group_by, curves_by, rows_by]).agg({
        metric: "median",
        "x_value": "median"
    }).reset_index().drop(columns=group_by)
    outdf[cols_by] = display_name

    full_df_list.append(outdf)

full_df = pd.concat(full_df_list)


# performance metric
rows = ["uniform"]
cols = full_df[cols_by].unique()
curves = full_df[curves_by].unique()

# plots
plt.cla()
fig, axs = plt.subplots(figsize=(8, 5), nrows=len(rows), ncols=len(cols),
                        sharex="col", sharey="row", squeeze=False)
for i, row in enumerate(rows):
    for j, col in enumerate(cols):
        # metric
        ax = axs[i, j]
        for _, curve in enumerate(curves):
            df = full_df.loc[(full_df[rows_by] == row) & (full_df[cols_by] == col) & (full_df[curves_by] == curve)]
            df = df.sort_values(by="x_value")
            ax.plot(df["x_value"], np.sqrt(df[metric]),
                    label=methods[curve][0], color=methods[curve][1],
                    linestyle=methods[curve][2], marker=methods[curve][3],
                    markerfacecolor='none')
            # if i == len(rows)-1:
            #     ax.set_xlabel(col)
            if i == 0:
                ax.set_title(f"Setting {'ABCDEFGHI'[j]}")
            for name, (group_by, display_var, display_name, logx) in experiments.items():
                if col == display_name:
                    ax.set_xscale("log" if logx else "linear")
            if j == 0:
                ax.set_ylabel(yaxis)
            if j == len(cols)-1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"{missing_mechanisms[row]}", rotation=270, labelpad=15)
        ax.set_yscale("log")

        for name, (group_by, display_var, display_name, logx) in experiments.items():
            if col == display_name:
                ax.set_xscale("log" if logx else "linear")
        if i == len(rows)-1:
            ax.set_xlabel(col)
        # ax.set_yscale("log")

# legend
lines = [Line2D([0], [0], color=color, linestyle=ltype, marker=mtype, markerfacecolor='none')
         for nm, (name, color, ltype, mtype) in methods.items() if nm in curves]
labels = [name for nm, (name, _, _, _) in methods.items() if nm in curves]

fig.legend(lines, labels, loc=9, ncol=5)
plt.tight_layout()
fig.subplots_adjust(top=0.8)
plt.savefig("experiments/synthetic_cputime.pdf")



