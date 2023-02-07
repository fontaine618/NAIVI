import numpy as np
from pypet import Environment, cartesian_product, Trajectory, Parameter, ParameterGroup
from pypet_experiments.data import Dataset
from pypet_experiments.method import Method
from pypet_experiments.results import Results
from pypet_experiments.run import run
from pypet_experiments.utils import add_parameters
from pypet_experiments.gather import gather_results


env = Environment(
    trajectory="test",
    filename="results/example/test.hdf5",
    overwrite_file=True,
    multiproc=False,
    ncores=0,
)
traj = env.trajectory
add_parameters(traj)

traj.f_explore(cartesian_product({
    "data.dataset": ["synthetic"],
    "data.n_nodes": [100],
    "data.p_cts": [0],
    "data.p_bin": [50],
    "data.seed": np.arange(0, 10).tolist(),
    "data.latent_dim": [5],
    "data.latent_variance": [1.],
    "data.latent_mean": [0.],
    "data.heterogeneity_variance": [1.],
    "data.heterogeneity_mean": [-2.],
    "data.cts_noise": [1.],
    "data.missing_covariate_rate": [0.5],
    "data.missing_edge_rate": [0.],
    "model.latent_dim": [5],
    "model.heterogeneity_prior_mean": [-2.],
    "model.heterogeneity_prior_variance": [1.],
    "model.latent_prior_mean": [0.],
    "model.latent_prior_variance": [1.],
    "method": ["VMP"],
    "fit.max_iter": [1000],
    "fit.rel_tol": [1e-5]
}))

env.run(run)

# debug
traj.f_set_crun(0)
traj.f_load(load_results=2)
traj.res.crun.f_to_dict()
traj.res.crun.logs.elbo_history

gather_results(traj)
gather_parameters_to_DataFrame(traj)["data.seed"]