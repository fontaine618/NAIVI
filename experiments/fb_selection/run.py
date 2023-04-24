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
        trajectory="fb_selection",
        filename=f"./results/seed{seed}.hdf5",
        overwrite_file=True,
        multiproc=True,
        ncores=1,
        use_pool=False,
    )
    traj = env.trajectory
    add_parameters(traj)

    traj.f_explore(cartesian_product({
        "data.dataset": ["facebook"],
        "data.path": ["~/Documents/NAIVI/datasets/facebook/"],
        "data.seed": [int(seed)],
        "data.facebook_center": [3980, 398, 686, 414, 348, 0, 3437, 1912, 1684, 107],
        "data.missing_covariate_rate": [0.5],
        "data.missing_edge_rate": [0.],
        "data.missing_mechanism": ["triangle"],
        "model.latent_dim": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "model.heterogeneity_prior_mean": [-2.],
        "model.heterogeneity_prior_variance": [1.],
        "model.latent_prior_mean": [0.],
        "model.latent_prior_variance": [1.],
        "method": ["VMP", ],
        "fit.vmp.max_iter": [100],
        "fit.vmp.rel_tol": [1e-5],
        "fit.vmp.cv_folds": [5],
    }))

    env.run(run)



