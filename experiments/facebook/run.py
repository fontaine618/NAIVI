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
        trajectory="facebook",
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
        "data.path": ["~/work/NAIVI/datasets/facebook/"],
        "data.seed": [int(seed)],
        "data.facebook_center": [3980, 698, 686, 414, 348, 0, 3437, 1912, 1684, 107],
        "data.missing_covariate_rate": [0.5],
        "data.missing_mechanism": ["triangle", "row_deletion"],
        "model.latent_dim": [0], # take the best selected (hard coded)
        "model.heterogeneity_prior_mean": [float("nan")],  # EB
        "model.heterogeneity_prior_variance": [float("nan")],
        "method": ["Mean", "VMP", "VMP0", "MAP", "MLE", "NetworkSmoothing", "MICE", "KNN", "FA"],
        "fit.vmp.max_iter": [1000],
        "fit.vmp.damping": [0.6],
    }))

    env.run(run)



