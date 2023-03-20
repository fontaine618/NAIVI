import torch
from pypet import cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np


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

results.reset_index(inplace=True, drop=True)
groupby = ["data.p_bin", "data.n_nodes"]

# compute new variables
gr2 = groupby + ["data.seed", "data.latent_dim"]

# results["model.df"] = results["model.latent_dim"] * results["data.p_bin"]
# results["data.log_n_nodes"] = np.log(results["data.n_nodes"].values.astype(float))
# results["data.loglog_n_nodes"] = np.log(np.log(results["data.n_nodes"].values.astype(float)))
# results["training.aic"] = -2*results["training.elbo"] + 2 * results["model.df"]
# results["training.bic"] = -2*results["training.elbo"] + results["model.df"] * results["data.log_n_nodes"]
# results["training.gic"] = -2*results["training.elbo"] + results["model.df"] * \
#     results["data.log_n_nodes"] * results["data.loglog_n_nodes"]

results["training.aic.rel_best"] = \
    results.groupby(gr2).apply(
        lambda x: x["training.aic"] / x["training.aic"].min()
    ).reset_index().drop(columns=gr2).set_index("level_4")
results["training.bic.rel_best"] = \
    results.groupby(gr2).apply(
        lambda x: x["training.bic"] / x["training.bic"].min()
    ).reset_index().drop(columns=gr2).set_index("level_4")
results["training.gic.rel_best"] = \
    results.groupby(gr2).apply(
        lambda x: x["training.gic"] / x["training.gic"].min()
    ).reset_index().drop(columns=gr2).set_index("level_4")
results["training.elbo.rel_best"] = \
    results.groupby(gr2).apply(
        lambda x: x["training.elbo"] / x["training.elbo"].max()
    ).reset_index().drop(columns=gr2).set_index("level_4")

K = 5
subset = results["data.latent_dim"]==K
rows = {
    "training.elbo.rel_best": "ELBO",
    "training.aic.rel_best": "AIC",
    "training.bic.rel_best": "BIC",
    "training.gic.rel_best": "GIC",
    "estimation.latent_Proj_fro_rel": r"$d(Z,\widehat{Z})$",
    "testing.X_bin_missing_auroc": "AuROC (missing)",
}
xvar = "model.latent_dim"
xdisplay = "Latent Dimension"
cols = groupby
title = f"Experiment 4: Model selection for missing binary covariates (true $K={K}$)"


df = results[subset]
df = df[[xvar] + list(rows.keys()) + cols + ["data.seed"]]
df = df.melt(id_vars=[xvar, *cols, "data.seed"], value_vars=rows.keys(), var_name="Metric", value_name="Value")
df["Metric"] = df["Metric"].replace(rows)
df["data.n_nodes"] = df["data.n_nodes"].replace({"100": "$N=100$", "1000": "$N=1,000$"})
df["data.p_bin"] = df["data.p_bin"].replace({"50": "$p=50$", "200": "$p=200"})
df["experiment"] = df["data.n_nodes"].astype(str) + ", " + df["data.p_bin"].astype(str)






plt.figure(figsize=(6, 8))
grid = sns.relplot(
    data=df,
    x=xvar,
    y="Value",
    col="experiment",
    row="Metric",
    hue="data.seed",
    kind="line",
    estimator="median",
    errorbar=("pi", 100),
    facet_kws={"sharey": "row", "sharex": True, "margin_titles": True},
    markers=True,
)
# grid.set(xscale="log")
grid.set_axis_labels(xdisplay, "")
# grid.axes[0, 0].set_ylim(1, 2)
# grid.set_titles("")
# for row, row_label in enumerate(row_labels):
#     grid.axes[row, 0].set_ylabel(row_label)
# grid.axes[0, :].set_titles(template="N, P={col_name}")
grid.set_titles(
    row_template="",
    col_template="N, P = {col_name}",
)
for row, row_label in enumerate(rows.values()):
    grid.axes[row, 0].set_ylabel(row_label)
grid.refline(x=K, linestyle="dashed", color="black")
grid.axes[4, 0].set_yscale("log")
grid.fig.tight_layout(w_pad=1)
grid.fig.subplots_adjust(top=0.95)
grid.fig.suptitle(title, y=0.98, x=grid.axes[0, 0].get_position().x0, horizontalalignment="left")
plt.savefig(f"experiments/{name}/figures/metrics_K{K}.pdf")




