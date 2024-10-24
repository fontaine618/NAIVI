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

name = "cora_selection"
res_list = []
for i in range(10):
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    res_list.append(results)
cora = pd.concat(res_list)
cora["training.elbo_plus_entropy"] = cora["training.elbo"] - cora["training.weights_entropy"]
cora["dataset"] = "cora"

name = "email_selection"
res_list = []
for i in range(10):
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    res_list.append(results)
email = pd.concat(res_list)
email["training.elbo_plus_entropy"] = email["training.elbo"] - email["training.weights_entropy"]
email["dataset"] = "email"

results = pd.concat([cora, email])

metrics = { # colname: (display_name, higher_is_better)
    "training.elbo": ("ELBO", True),
    "training.elbo_plus_entropy": ("ELBO - KL(B)", True),
    "testing.f1_multiclass_weighted": ("F1 (weighted)", True),
}

columns = { # name: xrange, xticks, display
    "cora": ((5, 10), [6, 8, 10], "Cora"),
    "email": ((3, 10), [4, 6, 8, 10], "Email"),
}
n_seeds = 5


fig, axs = plt.subplots(
    len(metrics), len(columns),
    figsize=(5*len(columns), 2*len(metrics)),
    sharey="row",
    sharex="col",
    squeeze=False
)

for col, (dataset, (xrange, xticks, display)) in enumerate(columns.items()):
    res_exp = results.loc[(results["data.n_seeds"] == n_seeds) & (results["dataset"] == dataset)]
    for row, (metric, (metric_name, higher_is_better)) in enumerate(metrics.items()):
        ax = axs[row, col]

        res_exp_metric = res_exp[[metric, "data.seed", "model.latent_dim"]]

        for i in res_exp_metric["data.seed"].unique():
            xs = res_exp_metric.loc[res_exp_metric["data.seed"] == i]["model.latent_dim"]
            ys = res_exp_metric.loc[res_exp_metric["data.seed"] == i][metric]
            val = (ys.max() if higher_is_better else ys.min())
            ys = (ys-val)/val
            ax.plot(xs, ys, marker="none", linestyle="solid", color="black", alpha=0.2)
            whichmin = ys.abs().values.argmin()
        ax.set_xticks(xticks)
        ax.set_xlim(xrange)

        if row == len(metrics)-1:
            ax.set_xlabel("$K$")
        if row == 0:
            ax.set_title(display)
        if row == 0 or row == 1:
            ax.set_ylim(0., 0.07)
        if col == 0:
            ax.set_ylabel(metric_name)
plt.tight_layout()
plt.savefig(f"./experiments/ss_selection_metrics.pdf")