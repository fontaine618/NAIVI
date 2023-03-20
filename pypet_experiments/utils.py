from pypet import Trajectory


_default_data_parameters = {
    "dataset": "synthetic",
    "n_nodes": 100,
    "p_cts": 0,
    "p_bin": 100,
    "seed": 0,
    "latent_dim": 5,
    "latent_variance": 1.,
    "latent_mean": 0.,
    "heterogeneity_variance": 1.,
    "heterogeneity_mean": -2.,
    "cts_noise": 1.,
    "missing_covariate_rate": 0.5,
    "missing_edge_rate": 0.,
}

_default_model_parameters = {
    "latent_dim": 5,
    "heterogeneity_prior_mean": -2.,
    "heterogeneity_prior_variance": 1.,
    "latent_prior_mean": 0.,
    "latent_prior_variance": 1.,
    "mnar": False,
    "regularization": 0.
}

_default_fit_parameters = {
    "vmp.max_iter": 1000,
    "vmp.rel_tol": 1e-4,
    "map.lr": 0.01,
    "map.max_iter": 1000,
    "map.eps": 1e-4,
    "map.optimizer": "Rprop",
    "advi.lr": 0.01,
    "advi.max_iter": 1000,
    "advi.eps": 1e-4,
    "advi.optimizer": "Rprop",
}


def add_parameters(traj: Trajectory):
    _add_data_parameters(traj)
    _add_model_parameters(traj)
    _add_method_parameters(traj)
    _add_fit_parameters(traj)


def _add_data_parameters(traj: Trajectory):
    for k, v in _default_data_parameters.items():
        traj.f_add_parameter(f"data.{k}", v)


def _add_model_parameters(traj: Trajectory):
    for k, v in _default_model_parameters.items():
        traj.f_add_parameter(f"model.{k}", v)


def _add_method_parameters(traj: Trajectory):
    traj.f_add_parameter("method", "VMP")


def _add_fit_parameters(traj: Trajectory):
    for k, v in _default_fit_parameters.items():
        traj.f_add_parameter(f"fit.{k}", v)