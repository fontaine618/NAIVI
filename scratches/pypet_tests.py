import numpy as np
import torch
import sys
import os
sys.path.insert(1, '/home/simfont/Documents/NAIVI/')
from pypet import Environment, cartesian_product
from pypet_experiments.run import run
from pypet_experiments.utils import add_parameters
from pypet_experiments.gather import gather_results_to_DataFrame, gather_parameters_to_DataFrame



env = Environment(
    trajectory="pypet_test",
    overwrite_file=True,
    multiproc=False,
)
traj = env.trajectory
add_parameters(traj)

traj.f_explore(cartesian_product({
    # "data.dataset": ["synthetic"],
    "data.dataset": ["facebook"],
    "data.facebook_center": [107],
    "data.path": ["/home/simon/Documents/NAIVI/datasets/facebook/"],
    # "data.seed": [0,],
    "data.seed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    # "data.n_seeds": [5],

    # "data.n_nodes": [500],
    # "data.p_cts": [0],
    # "data.p_bin": [20],
    # "data.latent_dim": [5],
    # "data.latent_dim_attributes": [4],
    # "data.latent_variance": [1.],
    # "data.latent_mean": [0.],
    # "data.heterogeneity_mean": [-2.],
    # "data.heterogeneity_variance": [1.],
    # "data.cts_noise": [1.],
    "data.missing_covariate_rate": [0.5],
    "data.missing_edge_rate": [0.],
    "data.missing_mechanism": ["row_deletion"],
    "model.latent_dim": [0, ],
    # "model.latent_dim": [2, 3, 4, 5, 6, 7, 8, 9],
    "model.heterogeneity_prior_mean": [float("nan")],
    "model.heterogeneity_prior_variance": [2.],
    "model.latent_prior_mean": [0.],
    "model.latent_prior_variance": [1.],
    "model.vmp.logistic_approximation": ["quadratic"],
    "method": [ "VMP"],
    # "method": [ "VMP",  "MICE", "Mean", "KNN", "FA", "MAP", "MLE", "NetworkSmoothing", "Oracle"],
    "fit.vmp.max_iter": [500, ],
    "fit.vmp.rel_tol": [1e-5],
    "fit.vmp.cv_folds": [0],
    # "fit.mice.max_iter": [50],
    # "fit.mice.rel_tol": [1e-3],
    # "model.gcn.n_hidden": [16],
    # "fit.gcn.weight_decay": [5e-4],
    # "fit.gcn.lr": [0.01],
    # "fit.gcn.max_iter": [500, ],
}))

env.run(run)

parameters = gather_parameters_to_DataFrame(traj)
results = gather_results_to_DataFrame(traj)
results = parameters.join(results)


# print(results[["method", "testing.mse_continuous"]].values)
print(results[["method", "testing.auroc_binary"]])