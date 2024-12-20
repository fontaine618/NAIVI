from pypet import Trajectory


_default_data_parameters = {
    "dataset": "synthetic",
    "facebook_center": 0,
    "path": "",
    "n_nodes": 100,
    "p_cts": 0,
    "p_bin": 100,
    "seed": 0,
    "latent_dim": 5,
    "latent_variance": 1.,
    "latent_mean": 0.,
    "latent": "continuous",
    "discrete_latent_components": 10,
    "heterogeneity_variance": 1.,
    "heterogeneity_mean": -2.,
    "cts_noise": 1.,
    "missing_covariate_rate": 0.5,
    "missing_edge_rate": 0.,
    "missing_mechanism": "uniform",
    "n_seeds": 1,
    "latent_dim_attributes": 0,
    "attribute_model": "inner_product",
    "edge_model": "inner_product"
}

_default_model_parameters = {
    "latent_dim": 5,
    "latent_dim_gb": 0,
    "heterogeneity_prior_mean": -2.,
    "heterogeneity_prior_variance": 1.,
    "latent_prior_mean": 0.,
    "latent_prior_variance": 1.,
    "mnar": False,
    "regularization": 0.,
    "vmp.logistic_approximation": "mk",
    "vmp.logistic_elbo": "quadrature",
    "vmp.init_precision": 0.,
    "gcn.n_hidden": 16,
    "gcn.dropout": 0.5,
    "network_weight": 1.
}

_default_fit_parameters = {
    "vmp.max_iter": 500,
    "vmp.min_iter": 20,
    "vmp.rel_tol": 1e-5,
    "vmp.cv_folds": 0,
    "vmp.damping": 0.6,
    "map.lr": 0.01,
    "map.max_iter": 1000,
    "map.eps": 1e-5,
    "map.optimizer": "Rprop",
    "mle.lr": 0.01,
    "mle.max_iter": 1000,
    "mle.eps": 1e-5,
    "mle.optimizer": "Rprop",
    "mice.max_iter": 20,
    "mice.rel_tol": 1e-3,
    "knn.n_neighbors": 10,
    "gcn.lr": 0.01,
    "gcn.max_iter": 500,
    "gcn.weight_decay": 0.001,
    "mcmc.num_samples": 2000,
    "mcmc.warmup_steps": 1000,
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