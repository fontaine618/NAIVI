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
    os.makedirs(f"results/", exist_ok=True)

    env = Environment(
        trajectory="cora",
        filename=f"./results/gcn2.hdf5",
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
        "data.seed": np.arange(0, 31).tolist(),
        "data.missing_edge_rate": [0.],
        "data.n_seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "model.latent_dim": [5],
        "model.heterogeneity_prior_mean": [float("nan")],
        "model.heterogeneity_prior_variance": [1.],
        "model.latent_prior_mean": [0.],
        "model.latent_prior_variance": [1.],
        "method": ["GCN"],
        "model.gcn.n_hidden": [16],
        "fit.gcn.weight_decay": [0.001],
        "fit.gcn.lr": [0.01],
        "fit.gcn.max_iter": [500],
    }))

    env.run(run)



