import numpy as np
import torch
import sys
import os
sys.path.insert(1, '/home/simfont/Documents/NAIVI/')
from pypet import Environment, cartesian_product
from pypet_experiments.run import run
from pypet_experiments.utils import add_parameters

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    seed = sys.argv[1]
    os.makedirs(f"results/", exist_ok=True)

    env = Environment(
        trajectory="missing_rate",
        filename=f"./results/seed{seed}.hdf5",
        overwrite_file=True,
        multiproc=True,
        ncores=1,
        use_pool=False,
    )
    traj = env.trajectory
    add_parameters(traj)

    traj.f_explore(cartesian_product({
        "data.dataset": ["synthetic"],
        "data.seed": [int(seed)],
        "data.n_nodes": [200],
        "data.p_cts": [0],
        "data.p_bin": [100],
        "data.latent_dim": [5],
        "data.latent_variance": [1.],
        "data.latent_mean": [0.],
        "data.heterogeneity_mean": [-2.],
        "data.heterogeneity_variance": [1.],
        "data.cts_noise": [1.],
        "data.missing_covariate_rate": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75],
        "data.missing_edge_rate": [0.],
        "data.missing_mechanism": ["row_deletion", "uniform", "triangle"],
        "model.latent_dim": [5],
        "model.heterogeneity_prior_mean": [-2.],
        "model.heterogeneity_prior_variance": [1.],
        "model.latent_prior_mean": [0.],
        "model.latent_prior_variance": [1.],
        "method": ["Mean", "KNN", "VMP", "FA", "MAP", "MLE", "ADVI", "NetworkSmoothing", "Oracle", "MICE"],
        "fit.vmp.max_iter": [100],
        "fit.vmp.rel_tol": [1e-5],
        "fit.vmp.cv_folds": [0],
    }))

    env.run(run)



