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

dataset = "cora"
display = "Cora"
name = "cora_selection"
xrange = (8, 15)
xticks = [8, 10, 12, 14, ]
n_seeds = [10, 20, 30]

# dataset = "email"
# display = "Email"
# name = "email_selection"
# xrange = (4, 12)
# xticks = [4, 6, 8, 10, 12]
# n_seeds = [3, 5, 8]

res_list = []
for i in range(10):
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)

    res_list.append(results)


results = pd.concat(res_list)
results["training.elbo_plus_entropy"] = results["training.elbo_exact"] - results["training.weights_entropy"]
results["dataset"] = dataset

metrics = { # colname: (display_name, higher_is_better, std)
    "training.elbo_exact": ("ELBO", True, True),
    "training.elbo_plus_entropy": ("ELBO - KL(B)", True, True),
    "testing.f1_multiclass_weighted": ("F1 (weighted)", True, False),
}


plt.gca()
fig, axs = plt.subplots(
    len(metrics), len(n_seeds),
    figsize=(10, 2*len(metrics)),
    sharey="row",
    sharex="col",
    squeeze=False
)

for col, n_seed in enumerate(n_seeds):
    res_exp = results.loc[(results["data.n_seeds"] == n_seed) & (results["dataset"] == dataset)]
    for row, (metric, (metric_name, higher_is_better, stadardize)) in enumerate(metrics.items()):
        ax = axs[row, col]

        res_exp_metric = res_exp[[metric, "data.seed", "model.latent_dim"]]

        for i in res_exp_metric["data.seed"].unique():
            xs = res_exp_metric.loc[res_exp_metric["data.seed"] == i]["model.latent_dim"].values.astype(float)
            ys = res_exp_metric.loc[res_exp_metric["data.seed"] == i][metric].values.astype(float)
            which = ~np.isinf(np.abs(ys)) * ~np.isnan(ys)
            if which.any():
                xs = xs[which]
                ys = ys[which]
                val = (ys.max() if higher_is_better else ys.min())
                if stadardize:
                    ys = (ys-val)/val
                ax.plot(xs, ys, marker="none", linestyle="solid", color="black", alpha=0.2)
        ax.set_xticks(xticks)
        ax.set_xlim(xrange)

        if row == len(metrics)-1:
            ax.set_xlabel("$K$")
        if row == 0:
            ax.set_title(f"{display}: {n_seed} seeds")
        # if row == 0 or row == 1:
        #     ax.set_ylim(0., 0.15)
        if row == 2:
            ax.set_ylim(0.6, 0.82)
        if col == 0:
            ax.set_ylabel(metric_name)
plt.tight_layout()
plt.savefig(f"./experiments/ss_selection_metrics_{dataset}.pdf")