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



name = "fb_selection"
res_list = []
for i in range(10):
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    results["data.seed"] = i
    res_list.append(results)

results = pd.concat(res_list)


results["training.elbo_plus_entropy"] = results["training.elbo_exact"] - \
                                        results["training.weights_entropy"]
results["training.elbo_covariates_plus_entropy"] = results["training.elbo_covariates"] - \
                                        results["training.weights_entropy"]

metrics = { # colname: (display_name, higher_is_better, rescale)
    "training.elbo_exact": ("ELBO", True, True),
    "training.elbo_plus_entropy": ("ELBO - H(B)", True, True),
    # "training.elbo_covariates": ("Cov. ELBO", True),
    # "training.elbo_covariates_plus_entropy": ("Cov. ELBO - H(B)", True),
    "testing.auroc_binary_weighted_average": ("AuROC", True, False),
}

cols = { #center :(N, p)
    # 3980: (59, 42),
    # 698: (66, 48),
    # 414: (159, 103),
    # 686: (170, 62),
    # 348: (227, 126),
    0: (347, 139),
    3437: (547, 116),
    1912: (755, 133),
    1684: (792, 100),
    107: (1045, 153)
}


fig, axs = plt.subplots(
    len(metrics), len(cols),
    figsize=(2*len(cols), 2*len(metrics)),
    sharey="row",
    sharex="col",
    squeeze=False,
)

for col, (center, (N, p)) in enumerate(cols.items()):
    res_exp = results.loc[results["data.facebook_center"] == center]
    for row, (metric, (metric_name, higher_is_better, rescale)) in enumerate(metrics.items()):
        ax = axs[row, col]

        res_exp_metric = res_exp[[metric, "data.seed", "model.latent_dim"]]

        for i in res_exp_metric["data.seed"].unique():
            xs = res_exp_metric.loc[res_exp_metric["data.seed"] == i]["model.latent_dim"]
            ys = res_exp_metric.loc[res_exp_metric["data.seed"] == i][metric]
            val = (ys.max() if higher_is_better else ys.min())
            if rescale:
                ys = (ys-val)/val
            ax.plot(xs, ys, marker="none", linestyle="solid", color="black", alpha=0.2)
            whichmin = ys.abs().values.argmin()
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        if row == len(metrics)-1:
            ax.set_xlabel("$K$")
        if row == 0:
            ax.set_title(f"{center} ($N={N}$, $p={p}$)")
        if col == 0:
            ax.set_ylabel(metric_name)
        if row == 0 or row == 1:
            ax.set_ylim(0., 0.07)
plt.tight_layout()

# plt.savefig(f"./experiments/{name}/fb_selection_metrics_small.pdf")
plt.savefig(f"./experiments/{name}/fb_selection_metrics_large.pdf")