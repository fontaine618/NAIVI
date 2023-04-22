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
        trajectory="cora",
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
        "data.path": ["~/Documents/NAIVI/datasets/cora/"],
        "data.seed": [int(seed)],
        "data.missing_edge_rate": [0.],
        "data.n_seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "model.latent_dim": [5],
        "model.heterogeneity_prior_mean": [float("nan")],
        "model.heterogeneity_prior_variance": [1.],
        "model.latent_prior_mean": [0.],
        "model.latent_prior_variance": [1.],
        "method": ["Mean", "KNN",  "VMP", "FA", "MAP", "MLE", "NetworkSmoothing", "MICE", "GCN"],
        "fit.vmp.max_iter": [100],
        "fit.vmp.rel_tol": [1e-5],
        "fit.vmp.cv_folds": [0],
        "model.gcn.n_hidden": [16],
        "fit.gcn.weight_decay": [0.001],
        "fit.gcn.lr": [0.01],
        "fit.gcn.max_iter": [1000],
    }))

    env.run(run)



