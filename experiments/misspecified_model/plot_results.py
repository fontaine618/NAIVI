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
    "misspecified_model": ("data.n_nodes", "data.n_nodes", "Nb. nodes", True),
}
seeds = range(30)

# Parameters
rows_by = "data.missing_mechanism"
curves_by = "method"
cols_by = ["data.attribute_model", "data.edge_model"]

models = {
    "inner_product": "Inner Product",
    "distance": "Distance",
}

# performance metric
metric = "testing.auroc_binary_weighted_average"
yaxis = "Pred. AuROC"


full_df_list = []
full_pdf_list = []

for name, (group_by, display_var, display_name, _) in experiments.items():

    res_list = []
    for i in seeds:
        file = f"./experiments/{name}/results/seed{i}.hdf5"
        tname = name+"_seed"+str(i)
        traj = Trajectory(name=tname)
        traj.f_load(filename=file, load_results=2, force=True)

        parameters = gather_parameters_to_DataFrame(traj)
        results = gather_results_to_DataFrame(traj)
        results = parameters.join(results)
        res_list.append(results)

    results = pd.concat(res_list)
    results["experiment"] = name


    cols = list(set([rows_by, curves_by, display_var, *cols_by, group_by, metric, "data.seed"]))
    df = results.loc[:, cols]
    df["x_value"] = df.groupby([group_by, rows_by])[display_var].transform("median")

    outdf = df.groupby([group_by, curves_by, rows_by, *cols_by]).agg({
        metric: "median",
        "x_value": "median"
    }).reset_index().drop(columns=group_by)

    pdf = pd.DataFrame(columns=[curves_by, rows_by, *cols_by, "x_value",
                                "p_value_two-sided", "stat_rwo-sided",
                                "p_value_less", "stat_less",
                                "p_value_greater", "stat_greater"])

    # wilcoxon siged rank test VMP vs others
    curves = df[curves_by].unique()
    curves = [c for c in curves if c != "VMP"]
    curves = [c for c in curves if c != "Oracle"]
    curves = [c for c in curves if isinstance(c, str)]
    rows = df[rows_by].unique()
    rows = [r for r in rows if isinstance(r, str)]
    cols0 = df[cols_by[0]].unique()
    cols1 = df[cols_by[1]].unique()
    xvals = df["x_value"].unique()
    xvals = [x for x in xvals if not math.isnan(x)]
    for meth, row, col0, col1, xvar in product(curves, rows, cols0, cols1, xvals):
        which_vmp = (df[curves_by] == "VMP") & (df[rows_by] == row) & (df[cols_by[0]] == col0) & (df[cols_by[1]] == col1) & (df["x_value"]==xvar)
        which_other = (df[curves_by] == meth) & (df[rows_by] == row) & (df[cols_by[0]] == col0) & (df[cols_by[1]] == col1) & (df["x_value"]==xvar)
        seeds_vmp = df.loc[which_vmp]["data.seed"].values
        seeds_other = df.loc[which_other]["data.seed"].values
        values_vmp = df.loc[which_vmp][metric].values
        values_other = df.loc[which_other][metric].values
        # subset to common seeds
        common_seeds = np.intersect1d(seeds_vmp, seeds_other)
        vmp = values_vmp[np.isin(seeds_vmp, common_seeds)]
        other = values_other[np.isin(seeds_other, common_seeds)]
        # make sure they are floats
        vmp = vmp.astype(float)
        other = other.astype(float)
        prop_better = np.mean(vmp < other)
        stat, p = wilcoxon(vmp, other, nan_policy="omit", alternative="two-sided")
        stat_l, p_l = wilcoxon(vmp, other, nan_policy="omit", alternative="less")
        stat_g, p_g = wilcoxon(vmp, other, nan_policy="omit", alternative="greater")
        # print(f"{meth} vs VMP on {row} {col} {xvar}: {prop_better:.2f} {p:.2e}")
        # store in pdf as new row
        pdf = pdf.append({
            rows_by: row,
            curves_by: meth,
            cols_by[0]: col0,
            cols_by[1]: col1,
            "x_value": xvar,
            "p_value_two-sided": p,
            "stat_rwo-sided": stat,
            "p_value_less": p_l,
            "stat_less": stat_l,
            "p_value_greater": p_g,
            "stat_greater": stat_g,
        }, ignore_index=True)

    full_df_list.append(outdf)
    full_pdf_list.append(pdf)

full_df = pd.concat(full_df_list)
full_pdf = pd.concat(full_pdf_list)

# 2x2 experiment
cols_by = ["data.attribute_model", "data.edge_model"]
full_df["models"] = [f"Attributes: {models[a]}\n Edges: {models[x]}" for a, x in zip(full_df[cols_by[0]], full_df[cols_by[1]])]
full_pdf["models"] = [f"Attributes: {models[a]}\n Edges: {models[x]}" for a, x in zip(full_pdf[cols_by[0]], full_pdf[cols_by[1]])]
cols_by = "models"

# performance metric
rows = full_df[rows_by].unique()
cols = full_df[cols_by].unique()
curves = full_df[curves_by].unique()
curves_pdf = full_pdf[curves_by].unique()

# plots
plt.cla()
fig, axs = plt.subplots(figsize=(12, 8), nrows=len(rows)*2, ncols=len(cols),
                        sharex="col", sharey="row", squeeze=False,
                        gridspec_kw={"height_ratios": [1, 0.4] * len(rows)})
for i, row in enumerate(rows):
    for j, col in enumerate(cols):
        # metric
        ax = axs[2*i, j]
        for _, curve in enumerate(curves):
            df = full_df.loc[(full_df[rows_by] == row) & (full_df[cols_by] == col) & (full_df[curves_by] == curve)]
            df = df.sort_values(by="x_value")
            ax.plot(df["x_value"], df[metric],
                    label=methods[curve][0], color=methods[curve][1],
                    linestyle=methods[curve][2], marker=methods[curve][3],
                    markerfacecolor='none')
            # if i == len(rows)-1:
            #     ax.set_xlabel(col)
            if i == 0:
                ax.set_title(col)
            if j == 0:
                ax.set_ylabel(yaxis)
            if j == len(cols)-1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"{missing_mechanisms[row]}", rotation=270, labelpad=15)
            # ax.set_ylim(0.95, 2.55)
        # wilcoxon p-values
        ax = axs[2*i+1, j]

        ax.axhline(y=np.log10(0.05), color="black", linestyle="--", alpha=0.5)
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.axhline(y=-np.log10(0.05), color="black", linestyle="--", alpha=0.5)
        for _, curve in enumerate(curves_pdf):
            pdf = full_pdf.loc[(full_pdf[rows_by] == row) & (full_pdf[cols_by] == col) & (full_pdf[curves_by] == curve)]
            pdf = pdf.sort_values(by="x_value")
            sign = (pdf["p_value_greater"] > pdf["p_value_less"])*1.
            y = sign * - np.log10(pdf["p_value_less"]) + (1. - sign) * np.log10(pdf["p_value_greater"])
            ax.plot(pdf["x_value"], y,
                    label=methods[curve][0], color=methods[curve][1],
                    linestyle="none", marker=methods[curve][3],
                    # linestyle=methods[curve][2], marker=methods[curve][3],
                    markerfacecolor='none')
            if i == len(rows)-1:
                ax.set_xlabel("Nb. nodes")
            # ax.set_yscale("log")
            if j == 0:
                ax.set_ylabel("Signed -log $p$-value \n NAIVI vs. Other")
        ax.set_xscale("log")
        ax.set_xticks([100, 200, 500], minor=False)
        ax.set_xticklabels([100, 200, 500], minor=False)
        ax.set_xticklabels([], minor=True)


# legend
lines = [Line2D([0], [0], color=color, linestyle=ltype, marker=mtype, markerfacecolor='none')
         for nm, (name, color, ltype, mtype) in methods.items() if nm in curves]
labels = [name for nm, (name, _, _, _) in methods.items() if nm in curves]

fig.legend(lines, labels, loc=9, ncol=9)
plt.tight_layout()
fig.subplots_adjust(top=0.88)
plt.savefig(f"experiments/{name}/results.pdf")



