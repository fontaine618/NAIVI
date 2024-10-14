import torch

from pypet import ParameterGroup, Parameter, Trajectory
from pypet_experiments.data import Dataset
from pypet_experiments.method import Method
from pypet_experiments.results import Results
from NAIVI.vmp import VMP, set_damping,set_check_args
torch.set_default_tensor_type(torch.cuda.FloatTensor)
# ================================================================================
# DATA SETTINGS
traj = Trajectory(name="test")
data_parms = {
    "dataset": "synthetic",
    "facebook_center": 0,
    "path": "",
    "n_nodes": 50,
    "p_cts": 20,
    "p_bin": 0,
    "seed": 0,
    "latent_dim": 2,
    "latent_variance": 1.,
    "latent_mean": 0.,
    "heterogeneity_variance": 1.,
    "heterogeneity_mean": -2.,
    "cts_noise": 1.,
    "missing_covariate_rate": 0.5,
    "missing_edge_rate": 0.,
    "missing_mechanism": "uniform",
    "n_seeds": 1,
    "latent_dim_attributes": 0,
    "attribute_model": "distance",
    "edge_model": "distance"
}
for k, v in data_parms.items():
    traj.f_add_parameter(f"data.{k}", data=v)

model_parms = {
    "latent_dim": 2,
    "heterogeneity_prior_mean": -2.,
    "heterogeneity_prior_variance": 1.,
    "latent_prior_mean": 0.,
    "latent_prior_variance": 1.,
    "mnar": False,
    "regularization": 0.,
    "vmp.logistic_approximation": "quadratic",
    "gcn.n_hidden": 16,
    "gcn.dropout": 0.5,
}
for k, v in model_parms.items():
    traj.f_add_parameter(f"model.{k}", data=v)
fit_parms = {
    "vmp.max_iter": 100,
    "vmp.rel_tol": 1e-5,
    "vmp.cv_folds": 0,
    "map.lr": 0.001,
    "map.max_iter": 2000,
    "map.eps": 1e-5,
    "map.optimizer": "Rprop",
    "mle.lr": 0.05,
    "mle.max_iter": 2000,
    "mle.eps": 1e-5,
    "mle.optimizer": "Rprop",
    "mice.max_iter": 20,
    "mice.rel_tol": 1e-3,
    "knn.n_neighbors": 10,
    "gcn.lr": 0.01,
    "gcn.max_iter": 200,
    "gcn.weight_decay": 5e-4,
    "mcmc.num_samples": 1000,
    "mcmc.warmup_steps": 500,
}
for k, v in fit_parms.items():
    traj.f_add_parameter(f"fit.{k}", data=v)
traj.f_add_parameter("method", data="VMP")
# ================================================================================


# ================================================================================
# RUN
traj.method = "Mean"


# get data instance (this could be loaded data or synthetic data)
data: Dataset = Dataset.from_parameters(traj.data)
# get method instance
method: Method = Method.from_parameters(traj.method, traj.model)
# run method on data and get results
results: Results = method.fit(data, traj.fit)


for k, v in results.to_dict().items():
    print(f"{k:<40}: {v}")


from pypet_experiments.method import _eb_heterogeneity_prior
import time
import math
# model_pararmeters = traj.model
# data_parameters = traj.data
# fit_parameters = traj.fit
# self = method
# covariates_only = False
#
# # set_damping(1.)
# self = vmp.factors["cts_model"]
from NAIVI.vmp.distributions import Normal
#
# vmp.factors["cts_observed"].values.values
# par=traj.data


