import torch

from pypet import Trajectory
from pypet_experiments.data import Dataset
from pypet_experiments.method import Method
from pypet_experiments.results import Results
torch.set_default_tensor_type(torch.cuda.FloatTensor)
# ================================================================================
# DATA SETTINGS
traj = Trajectory(name="test")
data_parms = {
    "dataset": "synthetic", # synthetic, email, facebook or cora
    "facebook_center": 3980,
    "n_seeds": 5, # for cora and email: number of seeds per class
    "path": "~/Documents/NAIVI/datasets/facebook/",
    "n_nodes": 500,
    "p_cts": 0,
    "p_bin": 100,
    "seed": 3,
    "latent_dim": 5,
    "latent_variance": 1.,
    "latent_mean": 0.,
    "latent": "continuous",
    "discrete_latent_components": 10,
    "heterogeneity_mean": -2.,
    "heterogeneity_variance": 1.,
    "cts_noise": 1.,
    "missing_covariate_rate": 0.5,
    "missing_edge_rate": 0.,
    "missing_mechanism": "uniform",
    "latent_dim_attributes": 0,
    "attribute_model": "inner_product",
    "edge_model": "inner_product"
}
for k, v in data_parms.items():
    traj.f_add_parameter(f"data.{k}", data=v)
# MODEL SETTINGS
model_parms = {
    "latent_dim": 5,
    # "heterogeneity_prior_mean": -2.,
    # "heterogeneity_prior_variance": 0.2,
    "heterogeneity_prior_mean": float("nan"), # nan for EB estimate
    "heterogeneity_prior_variance": float("nan"), # nan for EB estimate
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
# METHOD
traj.f_add_parameter("method", data="VMP")
# ESTIMATION SETTINGS
fit_parms = {
    "vmp.max_iter": 10000,
    "vmp.rel_tol": 1e-5,
    "vmp.cv_folds": 0,
    "vmp.damping": 0.7,
    "map.lr": 0.01,
    "map.max_iter": 1000,
    "map.eps": 1e-5,
    "map.optimizer": "Rprop",
    "mle.lr": 0.01,
    "mle.max_iter": 1000,
    "mle.eps": 1e-5,
    "mle.optimizer": "Rprop",
    "mice.max_iter": 10,
    "mice.rel_tol": 1e-3,
    "knn.n_neighbors": 10,
    "gcn.lr": 0.01,
    "gcn.max_iter": 500,
    "gcn.weight_decay": 5e-4,
    "mcmc.num_samples": 1000,
    "mcmc.warmup_steps": 500,
}
for k, v in fit_parms.items():
    traj.f_add_parameter(f"fit.{k}", data=v)
# ================================================================================




# ================================================================================
# RUN

# choose which method to run
# ["Mean", "VMP", "VMP0", "MAP", "MLE", "NetworkSmoothing", "MICE", "KNN", "FA", "Oracle", "MCMC", "GCN"]
# VMP0: VMP without heterogeneity
# MCMC is very slow, avoid more than 50 nodes/50 attributes
# GCN only works for the Cora dataset
traj.method = "VMP"


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
model_pararmeters = traj.model
data_parameters = traj.data
fit_parameters = traj.fit
self = method
covariates_only = False
heterogeneity = True


from NAIVI.vmp.messages import Message

# set_damping(1.)
# self = vmp.factors["cts_model"]
from NAIVI.vmp.distributions import Normal
#
# vmp.factors["cts_observed"].values.values
# par=traj.data


