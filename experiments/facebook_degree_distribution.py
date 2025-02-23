import torch
from pypet import cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import math
from scipy.stats import wilcoxon
from pypet import Trajectory
from pypet_experiments.data import Dataset
from pypet_experiments.method import Method
from pypet_experiments.results import Results

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

centers = { #center :(N, p)
    3980: (59, 42),
    698: (66, 48),
    414: (159, 103),
    686: (170, 62),
    348: (227, 126),
    0: (347, 139),
    3437: (547, 116),
    1912: (755, 133),
    1684: (792, 100),
    107: (1045, 153)
}

degrees = dict()

for c in centers.keys():

    # Load Cora
    traj = Trajectory(name="test")
    data_parms = {
        "dataset": "facebook", # synthetic, email, facebook or cora
        "path": "~/Documents/NAIVI/datasets/facebook/",
        "seed": 0,
        "missing_edge_rate": 0.,
        "missing_covariate_rate": 0.,
        "missing_mechanism": "uniform",
        "facebook_center": c
    }
    for k, v in data_parms.items():
        traj.f_add_parameter(f"data.{k}", data=v)
    data = Dataset.from_parameters(traj.data)

    A = torch.zeros((data.n_nodes, data.n_nodes))
    i0 = data.edge_index_left.long()
    i1 = data.edge_index_right.long()
    A[i0, i1] = data.edges.flatten()
    A[i1, i0] = data.edges.flatten()
    degrees[c] = A.sum(0).cpu().numpy()


# Plot
fig, axs = plt.subplots(2, 5, figsize=(10, 5), sharex=True, sharey=True)
for i, (c, (N, p)) in enumerate(centers.items()):
    ax = axs[i//5, i%5]
    sns.histplot(degrees[c]/N, ax=ax, kde=False, binwidth=0.05, binrange=(0, 1), stat="proportion")
    ax.set_title(f"Center {c}\n({N}, {p})")
    if i // 5 == 1:
        ax.set_xlabel("Degree/N")
    if i % 5 == 0:
        ax.set_ylabel("Proportion")
    else:
        ax.set_ylabel("")

plt.tight_layout()
plt.savefig(f"./experiments/facebook_degree_distribution.pdf")

for c in centers.keys():
    ds = degrees[c]
    mean = ds.mean()
    var = ds.var()
    print(f"Center {c}: mean={mean}, var={var}, var/mean={var/mean}")

