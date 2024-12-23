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



name = "k_selection"
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

results["training.elbo_plus_entropy"] = results["training.elbo"] - \
                                        results["training.weights_entropy"]
results["training.elbo_covariates_plus_entropy"] = results["training.elbo_covariates"] - \
                                        results["training.weights_entropy"]

metrics = { # colname: (display_name, higher_is_better, rescale)
    "training.elbo": ("ELBO", True, True),
    "training.elbo_plus_entropy": ("ELBO - H(B)", True, True),
    # "training.elbo_covariates": ("Cov. ELBO", True, True),
    # "training.elbo_covariates_plus_entropy": ("Cov. ELBO - H(B)", True, True),
    "testing.auroc_binary_weighted_average": ("AuROC", True, False),
}

cols = [ # N, p
    (200, 50),
    (200, 200),
    (500, 50),
    (500, 200),
]

K = 3
resK = results.loc[results["data.latent_dim"] == K]

fig, axs = plt.subplots(
    len(metrics), len(cols),
    figsize=(10, 2*len(metrics)),
    sharey="row",
    sharex="col"
)

for col, (N, p) in enumerate(cols):
    res_exp = resK.loc[
        (resK["data.n_nodes"] == N)
        & (resK["data.p_bin"] == p)
    ]
    for row, (metric, (metric_name, higher_is_better, rescale)) in enumerate(metrics.items()):
        ax = axs[row, col]

        res_exp_metric = res_exp[[metric, "data.seed", "model.latent_dim"]]

        for i in res_exp_metric["data.seed"].unique():
            xs = res_exp_metric.loc[res_exp_metric["data.seed"] == i]["model.latent_dim"]
            ys = res_exp_metric.loc[res_exp_metric["data.seed"] == i][metric]
            val = (ys.max() if higher_is_better else ys.min())
            ys = (ys-val)/val
            ax.plot(xs, ys, marker="none", linestyle="solid", color="black", alpha=0.2)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.axvline(x=K, linestyle="dashed", color="black", alpha=0.5)

        if row == len(metrics)-1:
            ax.set_xlabel("$K$")
            ax.set_ylim(-0.1, 0.01)
        if row in [0, 1]:
            ax.set_ylim(-0.01, 0.1)
        if row == 0:
            ax.set_title(f"$N={N}$, $p={p}$")
        if col == 0:
            ax.set_ylabel(metric_name)
plt.tight_layout()
fig.subplots_adjust(top=0.90)
# plt.show()
plt.suptitle(f"$K={K}$ latent dimensions", x=0.08,
             horizontalalignment='left')

plt.savefig(f"./experiments/{name}/metrics{K}.pdf")