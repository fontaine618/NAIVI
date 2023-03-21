import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/simon/Documents/NAIVI/")

plt.style.use("seaborn-v0_8-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import NAIVI
from old_stuff.NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.vmp import disable_logging
from NAIVI.vmp import VMP
from NAIVI.vmp.distributions import Distribution

# for debugging
from NAIVI.vmp.distributions.normal import MultivariateNormal

# to remove argument checking in distribution
# using -O also does this since this defualts to __debug__
Distribution.set_default_check_args(False)
NAIVI.vmp.enable_logging()
NAIVI.vmp.disable_logging()
NAIVI.vmp.set_check_args(0)

N = 30
p_bin = 20
p_cts = 0
K = 3

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, \
    i0, i1, A, B, B0, C, C0, W = \
    generate_dataset(
        N=N,
        K=K,
        p_cts=p_cts,
        p_bin=p_bin,
        var_cov=1.,
        missing_mean=0.,
        alpha_mean=-1.5,
        seed=0,
        mnar_sparsity=0.,
        adjacency_noise=0.,
        constant_components=True
    )

# Pyro model
import pyro
from pyro.distributions import MultivariateNormal, Normal, Unit, Categorical
from pyro.infer import MCMC, NUTS, Predictive
from pyro.infer.mcmc.util import initialize_model, summary

T = torch.Tensor

n_nodes = N
latent_dim = 3
edge_index_left = i0.reshape(-1)
edge_index_right = i1.reshape(-1)
edges = A.reshape(-1)
continuous_covariates = X_cts
binary_covariates = X_bin


def joint_latent_space_model(
        latent_dim: int,
        edge_index_left: T,
        edge_index_right: T,
        edges: T,
):
    n_nodes = max(edge_index_left.max() + 1, edge_index_right.max() + 1)
    n_edges = edge_index_left.shape[0]
    with pyro.plate("node", n_nodes):
        latent = pyro.sample("latent", MultivariateNormal(torch.zeros(latent_dim), torch.eye(latent_dim)))
        heterogeneity = pyro.sample("heterogeneity", Normal(-1.5, 1))
    with pyro.plate("node_pair", n_edges) as edge_index:
        theta_A = (latent[edge_index_left[edge_index]] * latent[edge_index_right[edge_index]]).sum(1) + \
                  heterogeneity[edge_index_left[edge_index]] + heterogeneity[edge_index_right[edge_index]]
        P = torch.sigmoid(theta_A)
        pyro.sample("edge", Categorical(P), obs=edges[edge_index])



init_params, potential_fn, transforms, _ = initialize_model(
    joint_latent_space_model,
    model_args=(latent_dim, edge_index_left, edge_index_right, edges),
    num_chains=3,
    jit_compile=True,
    skip_jit_warnings=False,
)

nuts_kernel = NUTS(potential_fn=potential_fn)

mcmc = MCMC(
    nuts_kernel,
    num_samples=100,
    warmup_steps=100,
    num_chains=3,
    initial_params=init_params,
    transforms=transforms,
)

mcmc.run(latent_dim, edge_index_left, edge_index_right, edges)

mcmc.get_samples()["latent"].mean(0)
alpha.flatten()

mcmc.diagnostics()