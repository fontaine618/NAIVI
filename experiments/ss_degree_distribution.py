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

# Load Cora
traj = Trajectory(name="test")
data_parms = {
    "dataset": "cora", # synthetic, email, facebook or cora
    "n_seeds": 40, # for cora and email: number of seeds per class
    "path": "~/Documents/NAIVI/datasets/cora/",
    "seed": 0,
    "missing_edge_rate": 0.,
}
for k, v in data_parms.items():
    traj.f_add_parameter(f"data.{k}", data=v)
cora = Dataset.from_parameters(traj.data)

A = torch.zeros((cora.n_nodes, cora.n_nodes))
i0 = cora.edge_index_left.long()
i1 = cora.edge_index_right.long()
A[i0, i1] = cora.edges.flatten()
A[i1, i0] = cora.edges.flatten()
cora_degrees = A.sum(0).cpu().numpy()


# Load Email
traj = Trajectory(name="test")
data_parms = {
    "dataset": "email", # synthetic, email, facebook or cora
    "n_seeds": 40, # for cora and email: number of seeds per class
    "path": "~/Documents/NAIVI/datasets/email/",
    "seed": 0,
    "missing_edge_rate": 0.,
}
for k, v in data_parms.items():
    traj.f_add_parameter(f"data.{k}", data=v)
cora = Dataset.from_parameters(traj.data)

A = torch.zeros((cora.n_nodes, cora.n_nodes))
i0 = cora.edge_index_left.long()
i1 = cora.edge_index_right.long()
A[i0, i1] = cora.edges.flatten()
A[i1, i0] = cora.edges.flatten()
email_degrees = A.sum(0).cpu().numpy()

# Plot
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
sns.histplot(cora_degrees, ax=ax[0], kde=False, binwidth=1, binrange=(0, 25))
ax[0].set_title("Cora")
ax[0].set_xlabel("Degree")
ax[0].set_ylabel("Frequency")
sns.histplot(email_degrees, ax=ax[1], kde=False, binwidth=5, binrange=(0, 250))
ax[1].set_title("Email")
ax[1].set_xlabel("Degree")
ax[1].set_ylabel("")
plt.tight_layout()
plt.savefig(f"./experiments/ss_degree_distribution.pdf")

