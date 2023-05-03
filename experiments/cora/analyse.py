import torch
from pypet import cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np


torch.set_default_tensor_type(torch.cuda.FloatTensor)
name = "cora"
res_list = []
for i in range(31):
    file = f"./experiments/{name}/results/seed{i}.hdf5"
    traj = Trajectory(name=name)
    traj.f_load(filename=file, load_results=2, force=True)

    parameters = gather_parameters_to_DataFrame(traj)
    results = gather_results_to_DataFrame(traj)
    results = parameters.join(results)
    res_list.append(results)

file = f"./experiments/{name}/results/gcn.hdf5"
traj = Trajectory(name=name)
traj.f_load(filename=file, load_results=2, force=True)

parameters = gather_parameters_to_DataFrame(traj)
results = gather_results_to_DataFrame(traj)
results = parameters.join(results)
res_list.append(results)

results = pd.concat(res_list)


sns.lineplot(
    data=results,
    x="data.n_seeds",
    y="testing.f1_multiclass_macro",
    hue="method",
)
plt.show()

