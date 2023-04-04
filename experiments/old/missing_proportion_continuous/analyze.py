import torch
from pypet import cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd


torch.set_default_tensor_type(torch.cuda.FloatTensor)
name = "missing_proportion_continuous"

res_list = []
for i in range(30):
    file = f"./experiments/missing_proportion_continuous/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    res_list.append(results)

results = pd.concat(res_list)




x_axis = "data.missing_covariate_rate"
rows = ["testing.X_cts_missing_mse", "estimation.latent_Proj_fro_rel"]
row_labels = ["MSE", r"$d(Z,\widehat{Z})$"]
hue = "method"
subset = results["data.p_cts"] > 0
cols = ["data.n_nodes", "data.p_cts"]
title = "Experiment 3: Missingness in continuous covariates\nVarying missing covariate rate"

# Automatic
df = results[subset]
df = df[[x_axis] + rows + [hue] + cols]
df = df.melt(id_vars=[x_axis, hue, *cols], value_vars=rows, var_name="Metric", value_name="Value")
df["Metric"] = df["Metric"].replace({x: y for x, y in zip(rows, row_labels)})
df["data.n_nodes"] = df["data.n_nodes"].replace({"100": "$N=100$", "1000": "$N=1,000$"})
df["data.p_cts"] = df["data.p_cts"].replace({"50": "$p=50$", "500": "$p=500$"})
df["experiment"] = df["data.n_nodes"].astype(str) + ", " + df["data.p_cts"].astype(str)
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


plt.figure(figsize=(6, 12))
grid = sns.relplot(
    data=df,
    x=x_axis,
    y="Value",
    hue=hue,
    col="experiment",
    row="Metric",
    kind="line",
    estimator="median",
    errorbar=("pi", 50),
    hue_order=["NAIVI-VMP", "NAIVI-QB", "MAP"],
    facet_kws={"sharey": "row", "sharex": True, "margin_titles": True},
    markers=True,
)
# grid.set(xscale="log")
grid.set_axis_labels("Missing rate", "")
grid.legend.set_title("Method")
grid.axes[0, 0].set_ylim(1, 2)
# grid.set_titles("")
# for row, row_label in enumerate(row_labels):
#     grid.axes[row, 0].set_ylabel(row_label)
# grid.axes[0, :].set_titles(template="N, P={col_name}")
grid.set_titles(
    row_template="",
    col_template="N, P = {col_name}",
)
for row, row_label in enumerate(row_labels):
    grid.axes[row, 0].set_ylabel(row_label)
grid.axes[1, 0].set_yscale("log")
grid.fig.tight_layout(w_pad=1)
grid.fig.subplots_adjust(top=0.85)
grid.fig.suptitle(title, y=0.95, x=grid.axes[0, 0].get_position().x0, horizontalalignment="left")
plt.savefig("experiments/missing_proportion_continuous/figures/metrics.pdf")



