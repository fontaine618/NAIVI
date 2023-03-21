import numpy as np
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import sys
import os
sys.path.insert(1, '/home/simfont/Documents/NAIVI/')
from pypet import Environment, cartesian_product
from pypet_experiments.run import run
from pypet_experiments.utils import add_parameters

seed = sys.argv[1]
os.makedirs(f"./results/", exist_ok=True)

env = Environment(
    trajectory="model_selection",
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
    "data.p_cts": [0],
    "data.p_bin": [50, 200],
    "data.seed": [int(seed)],
    "data.latent_dim": [3, 5],
    "data.latent_variance": [1.],
    "data.latent_mean": [0.],
    "data.heterogeneity_mean": [-2.],
    "data.heterogeneity_variance": [1.],
    "data.cts_noise": [1.],
    "data.missing_covariate_rate": [0.5],
    "data.missing_edge_rate": [0.],
    "model.latent_dim": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "model.heterogeneity_prior_mean": [-2.],
    "model.heterogeneity_prior_variance": [1.],
    "model.latent_prior_mean": [0.],
    "model.latent_prior_variance": [1.],
    "method": ["VMP"],
    "fit.vmp.max_iter": [1000],
    "fit.vmp.rel_tol": [1e-5],
    "fit.vmp.cv_folds": [5],
}))

env.run(run)



