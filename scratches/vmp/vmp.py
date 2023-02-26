import torch
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/simon/Documents/NAIVI/")

plt.style.use("seaborn-v0_8-whitegrid")
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import NAIVI
from NAIVI_experiments.gen_data_mnar import generate_dataset
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


N = 100
p_bin = 5
p_cts = 0

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, \
	i0, i1, A, B, B0, C, C0, W = \
	generate_dataset(
		N=N,
		K=3,
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

# compute some intermediary things
theta_X = Z @ B
ZtZ = (Z[i0, :] * Z[i1, :]).sum(1, keepdim=True)
theta_A = ZtZ + alpha[i0] + alpha[i1]
P = torch.sigmoid(theta_A)
cts_noise = torch.ones(p_cts)

true_values = {
	"heterogeneity": alpha,
	"latent": Z,
	"bias": B0,
	"weights": B,
	"cts_noise": cts_noise,
	"Theta_X": theta_X,
	"Theta_A": theta_A,
	"P": P,
	"X_cts": X_cts,
	"X_bin": X_bin,
	"X_cts_missing": X_cts_missing,
	"X_bin_missing": X_bin_missing,
	"A": A,
}



model = VMP(
	n_nodes=N,
	binary_covariates=X_bin,
	continuous_covariates=X_cts,
	edges=A,
	# edges=None,
	edge_index_left=i0,
	edge_index_right=i1,
	latent_dim=3,
	heterogeneity_prior_mean=-1.5
)
# model.fit(max_iter=100, rel_tol=1e-7)


model.fit_and_evaluate(
	max_iter=100,
	true_values=true_values
)

Best = model.factors["affine_bin"].parameters["weights"]
B.T @ B
Best.T @ Best

print(model.metrics_history["X_bin_missing_auroc"][-1])
print(max(model.metrics_history["X_bin_missing_auroc"]))


import cProfile
cProfile.run("model.fit(max_iter=100)")


self = model.factors["bin_model"]
self._quadratic_elbo()
self._quadrature_elbo()
self._mk_elbo()
self._tilted_elbo()
self.elbo_mc(100)

from NAIVI.vmp.factors.logistic_utls import *

m = torch.linspace(-5, 5, 11).unsqueeze(-1).expand(11, -1)

elbo[elbo!=0.]
elbo = elbo.mean(0)
elbo[~elbo.isnan()]
s[s.abs()==0.5]

fn = lambda x: x * torch.sigmoid(x)
_gh_quadrature(m, v, fn)[:10, 0]
m1[:10, 0]
_ms_expit_moment(1, m, v)[:10, 0]



self = model.factors["cts_model"]
#
#
# # plot factor graphs
# import networkx as nx
# factors = self.factors
# variables = self.variables
# messages = self.messages
#
# G = nx.DiGraph()
# for factor in factors.values():
# 	G.add_node(repr(factor))
# 	G.nodes[repr(factor)]["shape"] = "s"
# for variable in variables.values():
# 	G.add_node(repr(variable))
# for message in messages:
# 	G.add_edge(repr(message.factor), repr(message.variable), label=repr(message))
#
#
# plt.figure(figsize=(20, 20))
# layout = nx.planar_layout(G)
# nx.draw_networkx_nodes(G, node_shape="s", nodelist=[repr(factor) for factor in factors.values()],
# 					   pos=layout, node_color="k")
# nx.draw_networkx_nodes(G, node_shape="o", nodelist=[repr(variable) for variable in variables.values()],
# 					   pos=layout, node_color="w", edgecolors="k")
# nx.draw_networkx_edges(G, pos=layout, arrows=False)
# nx.draw_networkx_edge_labels(G, pos=layout, label_pos=0.5, edge_labels=nx.get_edge_attributes(G, "label"))
# plt.show()






#
# # ELBO History plot
# df = pd.DataFrame(vmp.elbo_history)
# df = df.loc[:, df.var() > 0.]
# df.plot()
# plt.xscale("log")
# plt.title("ELBOs")
# plt.show()
#
# # ELBO MC History plot
# df = pd.DataFrame(vmp.elbo_mc_history)
# df = df.loc[:, df.var() > 0.]
# df.plot()
# plt.xscale("log")
# plt.title("ELBOs (MC)")
# plt.show()
#
# # Metric plot
# df = pd.DataFrame(vmp.metrics_history["X_bin_missing_auroc"])
# df = df.loc[:, df.var() > 0.]
# df.plot()
# plt.xscale("log")
# plt.show()



#
#
# # For experiment
#
# N = 200
# K = 3
# p_cts = 50
# p_bin = 50
# p = p_cts + p_bin
# var_cov = 1.
# missing_mean = 0.
# seed = 0
# alpha_mean_gen = -1.85
# mnar_sparsity = 0.
# adjacency_noise = 0.
# constant_components = True
# K_model = 3
# mnar = False
# alpha_mean_model = 0.
# reg = 0.
# network_weight = 1.
# estimate_components = False
# algo = "VMP"
# max_iter = 100
# n_samples = 0
# eps = 1e-5
# keep_logs = True
# power = 1.
# init_method = ""
# optimizer = "RProp"
# lr = 0.01
# mcmc_n_sample = 0
# mcmc_n_chains = 0
# mcmc_n_warmup = 0
# mcmc_n_thin = 0
#
#
