import numpy as np
import torch
from pypet import Environment, cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.data import Dataset
from pypet_experiments.method import Method
from pypet_experiments.results import Results
from pypet_experiments.run import run
from pypet_experiments.utils import add_parameters
from pypet_experiments.gather import gather_parameters_to_DataFrame, gather_results_to_DataFrame


torch.set_default_tensor_type(torch.cuda.FloatTensor)
env = Environment(
    trajectory="test",
    filename="./results/example/test.hdf5",
    overwrite_file=True,
    multiproc=True,
    ncores=2,
)
traj = env.trajectory
add_parameters(traj)

traj.f_explore(cartesian_product({
    "data.dataset": ["synthetic"],
    "data.n_nodes": [100],
    "data.p_cts": [0],
    "data.p_bin": [50],
    "data.seed": np.arange(0, 10).tolist(),
    "data.latent_dim": [5],
    "data.latent_variance": [1.],
    "data.latent_mean": [0.],
    "data.heterogeneity_variance": [1.],
    "data.heterogeneity_mean": [-2.],
    "data.cts_noise": [1.],
    "data.missing_covariate_rate": [0.5],
    "data.missing_edge_rate": [0.],
    "model.latent_dim": [5],
    "model.heterogeneity_prior_mean": [-2.],
    "model.heterogeneity_prior_variance": [1.],
    "model.latent_prior_mean": [0.],
    "model.latent_prior_variance": [1.],
    "method": ["MLE", "FA"],
    "fit.vmp.max_iter": [1000],
    "fit.vmp.rel_tol": [1e-5],
    "fit.map.lr": [0.01],
    "fit.map.max_iter": [1000],
    "fit.map.eps": [1e-4],
    "fit.map.optimizer": ["Rprop"],
    "fit.advi.lr": [0.01],
    "fit.advi.max_iter": [1000],
    "fit.advi.eps": [1e-4],
    "fit.advi.optimizer": ["Rprop"],
}))

env.run(run)

traj.f_load(load_results=2)
res = gather_results_to_DataFrame(traj)
# gather_parameters_to_DataFrame(traj)["data.seed"]
print(res["testing.X_bin_missing_auroc"])





# # debug
traj.f_set_crun(0)
data: Dataset = Dataset.from_parameters(traj.data)
method: Method = Method.from_parameters(traj.method, traj.model)
results: Results = method.fit(data, traj.fit)
# import cProfile
# cProfile.run("method.fit(data, traj.fit)")
# # traj.res.crun.f_to_dict()
# # traj.res.crun.logs.elbo_history
#
method_parameters = traj.method
fit_parameters = traj.fit
self = method
#
#
# name = "weights"
# value = data.true_values[name]


traj.f_load(load_results=2)
res = gather_results_to_DataFrame(traj)
gather_parameters_to_DataFrame(traj)["data.seed"]
print(res["testing.X_bin_missing_auroc"])



import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

plt.cla()
for index, value in res["logs.elbo_history"].iteritems():
    value = np.array(value)
    i = np.argmax(np.diff(value) < 0.) + 1
    value = (max(value)-value)/max(abs(value))+1e-5
    plt.plot(value, label=index, alpha=0.2)
    plt.scatter(i, value[i], marker="x", label=index, s=100)
    plt.scatter(len(value), value[-1], marker=".", label=index, s=100)
    plt.scatter(np.argmin(value), min(value), marker="^", label=index, s=100)
plt.yscale("log")
# plt.xscale("log")
plt.tight_layout()
plt.show()