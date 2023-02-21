import torch
from pypet import cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd


torch.set_default_tensor_type(torch.cuda.FloatTensor)
name = "n_covariates_binary"

res_list = []
for i in range(30):
    file = f"./experiments/n_covariates_binary/results/results{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    res_list.append(results)

results = pd.concat(res_list)




x_axis = "data.p_bin"
rows = ["testing.X_bin_missing_auroc", "estimation.latent_Proj_fro_rel"]
row_labels = ["AuROC", r"$d(Z,\widehat{Z})$"]
hue = "method"
subset = results["data.p_bin"] > 0
cols = "data.n_nodes"
col_titles = ["$N=100$", "$N=1,000$"]
title = "Experiment 2: Missingness in binary covariates"

# Automatic
df = results[subset]
df = df[[x_axis] + rows + [hue] + [cols]]
df = df.melt(id_vars=[x_axis, hue, cols], value_vars=rows, var_name="Metric", value_name="Value")
df["Metric"] = df["Metric"].replace({x: y for x, y in zip(rows, row_labels)})
df["data.n_nodes"] = df["data.n_nodes"].replace({x: y for x, y in zip(cols, col_titles)})
df["method"] = df["method"].replace({
    "VMP": "NAIVI-VMP",
    "MAP": "MAP",
    "ADVI": "NAIVI-QB",
})



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


plt.figure(figsize=(8, 8))
grid = sns.relplot(
    data=df,
    x=x_axis,
    y="Value",
    hue=hue,
    col=cols,
    row="Metric",
    kind="line",
    estimator="median",
    errorbar=("pi", 50),
    hue_order=["NAIVI-VMP", "NAIVI-QB", "MAP"],
    # col_order=col_titles,
    # row_order=row_labels,
    facet_kws={"sharey": "row", "sharex": True},
    markers=True,
)
grid.set(xscale="log")
grid.set_axis_labels("Number of covariates", "")
grid.set_titles("")
grid.legend.set_title("Method")
for row, row_label in enumerate(row_labels):
    grid.axes[row, 0].set_ylabel(row_label)
for col, col_label in enumerate(col_titles):
    grid.axes[0, col].set_title(col_label)
grid.axes[1, 0].set_yscale("log")
grid.fig.tight_layout(w_pad=1)
grid.fig.subplots_adjust(top=0.90)
grid.fig.suptitle(title, y=0.95, x=grid.axes[0, 0].get_position().x0, horizontalalignment="left")
plt.savefig("experiments/n_covariates_binary/figures/metrics.pdf")



