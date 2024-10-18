import numpy as np
import torch
import sys
import os
sys.path.insert(1, '/storage/home/spf5519/work/NAIVI/')
# sys.path.insert(1, '/home/simon/Documents/NAIVI/')
from pypet import Environment, cartesian_product
from pypet_experiments.run import run
from pypet_experiments.utils import add_parameters

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    seed = sys.argv[1]
    os.makedirs(f"results/", exist_ok=True)

    env = Environment(
        trajectory=f"misspecified_model_seed{seed}",
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
        "data.n_nodes": [100, 200, 500],
        "data.p_cts": [0],
        "data.p_bin": [50],
        "data.latent_dim": [5],
        "data.latent_variance": [1.],
        "data.latent_mean": [0.],
        "data.heterogeneity_mean": [-2.],
        "data.heterogeneity_variance": [1.],
        "data.cts_noise": [1.],
        "data.missing_covariate_rate": [0.5],
        "data.missing_edge_rate": [0.],
        "data.missing_mechanism": ["uniform", "row_deletion"],
        "data.attribute_model": ["inner_product", "distance"],
        "data.edge_model": ["inner_product", "distance"],
        "model.latent_dim": [5],
        "model.heterogeneity_prior_mean": [-2.],
        "model.heterogeneity_prior_variance": [1.],
        "model.latent_prior_mean": [0.],
        "model.latent_prior_variance": [1.],
        "method": ["Mean", "KNN", "VMP", "FA", "MAP", "MLE", "NetworkSmoothing", "Oracle", "MICE"],
        "fit.vmp.max_iter": [100],
        "fit.vmp.rel_tol": [1e-5],
        "fit.vmp.cv_folds": [0],
        "mcmc.num_samples": [1000],
        "mcmc.warmup_steps": [500],
    }))

    env.run(run)


