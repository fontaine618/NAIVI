import numpy as np
import torch
import sys
sys.path.extend(["/home/simon/Documents/NAIVI/"])
print(sys.path)
from pypet import Environment, cartesian_product
from pypet_experiments.run import run
from pypet_experiments.utils import add_parameters


torch.set_default_tensor_type(torch.cuda.FloatTensor)
env = Environment(
    trajectory="n_covariates_binary",
    filename="./results.hdf5",
    overwrite_file=True,
    multiproc=True,
    ncores=1,
)
traj = env.trajectory
add_parameters(traj)

traj.f_explore(cartesian_product({
    "data.dataset": ["synthetic"],
    "data.n_nodes": [100, 1000],
    "data.p_cts": [0],
    "data.p_bin": [5, 10, 20, 50, 100, 200, 500, 1000],
    "data.seed": np.arange(0, 30).tolist(),
    "data.latent_dim": [5],
    "data.latent_variance": [1.],
    "data.latent_mean": [0.],
    "data.heterogeneity_mean": [-2.],
    "data.heterogeneity_variance": [1.],
    "data.cts_noise": [1.],
    "data.missing_covariate_rate": [0.5],
    "data.missing_edge_rate": [0.],
    "model.latent_dim": [5],
    "model.heterogeneity_prior_mean": [-2.],
    "model.heterogeneity_prior_variance": [1.],
    "model.latent_prior_mean": [0.],
    "model.latent_prior_variance": [1.],
    "method": ["MAP", "ADVI", "VMP"],
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



