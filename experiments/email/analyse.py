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


# Methods
methods = {
    # "Oracle":           ("Oracle",      "#000000", "solid", "s"),

    "VMP":              ("NAIVI-VMP",   "#1111ff", "solid", "o"),
    # "ADVI":             ("NAIVI-QB",    "#8888ff", "dashed", "v"),

    "MAP":              ("NAIVI-MAP",   "#3333ff", "dotted", "s"),
    # "MLE":              ("NAIVI-MLE",   "#3333ff", "dotted", "v"),
    "NetworkSmoothing": ("Smoothing",   "#8888ff", "dashed", "s"),

    # "FA":               ("FA",          "#88ffcc", "dotted", "o"),
    # "KNN":              ("KNN",         "#88ff88", "dashed", "v"),
    # "MICE":             ("MICE",        "#88ff88", "dashed", "s"),

    # "Mean":             ("Mean",        "#55cc55", "dotted", "s"),
}



name = "email"
res_list = []
for i in range(31):
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    res_list.append(results)

results = pd.concat(res_list)

metrics = {
    "testing.auroc_multiclass": ("Multiclass AuROC", ),
    "testing.f1_multiclass_micro": ("F1 (micro)", ),
    "testing.f1_multiclass_macro": ("F1 (macro)", ),
    "testing.accuracy_multiclass": ("Accuracy (%)", ),
}


fig, axs = plt.subplots(1, len(metrics), figsize=(10, 4), sharey=False)

for i, (metric, (metric_name, )) in enumerate(metrics.items()):
    ax = axs[i]

    for i, (method, (method_name, color, linestyle, marker)) in enumerate(methods.items()):
        res_method = results.loc[results["method"] == method]
        ys = res_method.groupby("data.n_seeds").agg(
            median=(metric, "median"),
            lower=(metric, lambda x: np.quantile(x, 0.25)),
            upper=(metric, lambda x: np.quantile(x, 0.75)),
        )
        xs = ys.index

        ax.plot(xs, ys["median"], color=color, linestyle=linestyle,
                marker=marker, label=method_name, markerfacecolor='none')
        ax.fill_between(xs, ys["lower"], ys["upper"], color=color, alpha=0.2)

    ax.set_title(f"{metric_name}")
    ax.set_xlabel("Seeds / Dpt.", size=10)
    ax.set_xticks([2, 4, 6, 8, 10])
# legend
lines = [Line2D([0], [0], color=color, linestyle=ltype, marker=mtype, markerfacecolor='none')
         for _, (_, color, ltype, mtype) in methods.items()]
labels = [name for _, (name, _, _, _) in methods.items()]
fig.legend(lines, labels, loc=9, ncol=6)
plt.tight_layout()
fig.subplots_adjust(top=0.80)
# plt.show()

plt.savefig(f"./experiments/{name}/metrics.pdf")