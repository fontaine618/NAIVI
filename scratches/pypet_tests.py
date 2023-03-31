import sys
sys.path.insert(1, '/home/simon/Documents/NAIVI/')
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
    "data.dataset": ["synthetic"],
    # "data.dataset": ["facebook"],
    # "data.facebook_center": [0],
    # "data.path": ["/home/simon/Documents/NAIVI/datasets/facebook/"],

    "data.n_nodes": [100],
    "data.p_cts": [0],
    "data.p_bin": [20],
    "data.seed": [0],
    "data.latent_dim": [3],
    "data.latent_variance": [1.],
    "data.latent_mean": [0.],
    "data.heterogeneity_mean": [-2.],
    "data.heterogeneity_variance": [1.],
    "data.cts_noise": [1.],
    "data.missing_covariate_rate": [0.5],
    "data.missing_edge_rate": [0.],
    "data.missing_mechanism": ["row_deletion"],
    "model.latent_dim": [3],
    "model.heterogeneity_prior_mean": [-2.],
    "model.heterogeneity_prior_variance": [1.],
    "model.latent_prior_mean": [0.],
    "model.latent_prior_variance": [1.],
    "method": ["KNN", "Mean", "NetworkSmoothing", "VMP", "MAP", "MLE", "ADVI", "FA", "Oracle"],
    "fit.vmp.max_iter": [200],
    "fit.vmp.rel_tol": [1e-5],
    "fit.vmp.cv_folds": [0],
}))

env.run(run)

parameters = gather_parameters_to_DataFrame(traj)
results = gather_results_to_DataFrame(traj)
results = parameters.join(results)

print(results[["method", "testing.X_bin_missing_auroc_multiclass"]])