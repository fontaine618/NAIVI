import numpy as np
import torch
import sys
sys.path.extend(["/home/simfont/Documents/NAIVI/"])
from pypet import Environment, cartesian_product
from pypet_experiments.run import run
from pypet_experiments.utils import add_parameters

seed = sys.argv[1]

torch.set_default_tensor_type(torch.cuda.FloatTensor)
env = Environment(
    trajectory="missing_proportion_continuous",
    filename=f"./results/seed{seed}.hdf5",
    overwrite_file=True,
    multiproc=True,
    ncores=1,
)
traj = env.trajectory
add_parameters(traj)

traj.f_explore(cartesian_product({
    "data.dataset": ["synthetic"],
    "data.n_nodes": [100, 1000],
    "data.p_cts": [50, 500],
    "data.p_bin": [0],
    "data.seed": [int(seed)],
    "data.latent_dim": [5],
    "data.latent_variance": [1.],
    "data.latent_mean": [0.],
    "data.heterogeneity_mean": [-2.],
    "data.heterogeneity_variance": [1.],
    "data.cts_noise": [1.],
    "data.missing_covariate_rate": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                    0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
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



