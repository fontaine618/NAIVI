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
        trajectory="cora_selection",
        filename=f"./results/seed{seed}.hdf5",
        overwrite_file=True,
        multiproc=True,
        ncores=1,
        use_pool=False,
    )
    traj = env.trajectory
    add_parameters(traj)

    traj.f_explore(cartesian_product({
        "data.dataset": ["cora"],
        "data.path": ["~/work/NAIVI/datasets/cora/"],
        "data.seed": [int(seed)],
        # "data.n_seeds": [3, 5, 8],
        "data.n_seeds": [3, 5, 8],
        "model.heterogeneity_prior_mean": [float("nan")],
        "model.heterogeneity_prior_variance": [float("nan")],
        "model.latent_prior_mean": [0.],
        "model.latent_prior_variance": [1.],
        # "model.latent_dim": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        "model.latent_dim": [2, 5, 7, 10, 12],
        "method": ["VMP", ],
        "fit.vmp.max_iter": [2000],
        "model.vmp.logistic_approximation": ["adaptive"],
        "model.vmp.logistic_elbo": ["quadrature"],
    }))

    env.run(run)



