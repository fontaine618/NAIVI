import torch
import math
import pyro
import pyro.distributions as dist
import graphviz

# =============================================================================
# generate data
n_nodes = 33
p_bin = 0
p_cts = 0
X_bin = torch.randint(0, 2, (n_nodes, p_bin))
M_bin = torch.randint(0, 3, (n_nodes, p_bin))
X_cts = torch.randn(n_nodes, p_cts)
M_cts = torch.randint(0, 3, (n_nodes, p_cts))
X_bin = torch.where(M_bin.eq(1), torch.tensor(float("nan")), X_bin)
X_cts = torch.where(M_cts.eq(1), torch.tensor(float("nan")), X_cts)
i = torch.tril_indices(n_nodes, n_nodes, offset=-1)
edge_index_left, edge_index_right = i[0, :], i[1, :]
n_edges = edge_index_left.size(0)
edges = torch.randint(0, 2, (n_edges,)).float()
binary_covariates = X_bin
continuous_covariates = X_cts

i_cts, j_cts = (~torch.isnan(continuous_covariates)).nonzero(as_tuple=True)
i_bin, j_bin = (~torch.isnan(binary_covariates)).nonzero(as_tuple=True)
x_cts = continuous_covariates[i_cts, j_cts]
x_bin = binary_covariates[i_bin, j_bin]
# -----------------------------------------------------------------------------






# =============================================================================
# model
from NAIVI.mcmc import MCMC
self = MCMC(
    n_nodes=n_nodes,
    latent_dim=3,
    binary_covariates=X_bin,
    continuous_covariates=X_cts,
    edges=edges,
    edge_index_left=edge_index_left,
    edge_index_right=edge_index_right,
)
self.fit(num_samples=1000, warmup_steps=200)


samples = self.get_samples_with_derived_quantities()
pred = self.predict()

self.output()
self.output_with_uncertainty()


from NAIVI.vmp import VMP

self = VMP(
    latent_dim=3,
    n_nodes=n_nodes,
    binary_covariates=X_bin,
    continuous_covariates=X_cts,
    edges=edges.unsqueeze(1),
    edge_index_left=edge_index_left,
    edge_index_right=edge_index_right
)

self.fit()

self.output()
self.output_with_uncertainty().keys()

c_id = vmp.variables["cts_obs"].id
vmp.factors["cts_model"].messages_to_children[c_id].message_to_variable.mean_and_variance
vmp.factors["cts_model"].parameters["log_variance"].exp()

self = vmp.factors["cts_model"]






samples = mcmc.get_samples_with_derived_quantities()
samples["llk"].reciprocal().mean().pow(-1.)
samples["llk"].mean()
(samples["llk"]+155.).exp().mean().log()-155.