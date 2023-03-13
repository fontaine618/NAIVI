import torch
from pypet import cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd


torch.set_default_tensor_type(torch.cuda.FloatTensor)
name = "model_selection"
res_list = []
for i in range(30):
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    res_list.append(results)

results = pd.concat(res_list)


x_axis = "model.latent_dim"
y_axes = ["testing.X_bin_missing_auroc", "estimation.latent_Proj_fro_rel", "training.cpu_time"]
y_labels = ["AuROC", r"$d(Z,\widehat{Z})$", "CPU Time"]
hue = "method"
subset = results["data.p_bin"]==100
title = "Experiment 1: Missingness in binary covariates"

# Automatic
df = results[subset]
df = df[[x_axis] + y_axes + [hue]]
df = df.melt(id_vars=[x_axis, hue], value_vars=y_axes, var_name="Metric", value_name="Value")
df["Metric"] = df["Metric"].replace({y_axis: y_label for y_axis, y_label in zip(y_axes, y_labels)})
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


plt.figure(figsize=(8, 4))
grid = sns.relplot(
    data=df,
    x=x_axis,
    y="Value",
    hue=hue,
    col="Metric",
    kind="line",
    errorbar="sd",
    hue_order=["NAIVI-VMP", "NAIVI-QB", "MAP"],
    facet_kws={"sharey": False, "sharex": True},
    markers=True,
)
grid.set(xscale="log")
grid.axes[0, 1].set_yscale("log")
grid.axes[0, 2].set_yscale("log")
grid.set_axis_labels("Number of nodes", "")
grid.set_titles("")
grid.legend.set_title("Method")
for ax, y_label in zip(grid.axes.flat, y_labels):
    ax.set_ylabel(y_label)
grid.fig.tight_layout(w_pad=1)
grid.fig.subplots_adjust(top=0.90)
grid.fig.suptitle(title, y=0.95, x=grid.axes[0, 0].get_position().x0, horizontalalignment="left")
plt.savefig("experiments/n_nodes/figures/metric.pdf")



