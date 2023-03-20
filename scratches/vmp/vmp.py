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


N = 200
p_bin = 0
p_cts = 300
K = 5

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

out = {}

for K_model in range(1, 11):

	model = VMP(
		n_nodes=N,
		binary_covariates=X_bin,
		continuous_covariates=X_cts,
		edges=A,
		# edges=None,
		edge_index_left=i0,
		edge_index_right=i1,
		latent_dim=K_model,
		heterogeneity_prior_mean=-1.5
	)
	# model.fit(max_iter=100, rel_tol=1e-7)


	model.fit_and_evaluate(
		max_iter=200,
		true_values=true_values
	)

	out[K_model] = model.elbo(), model.df, model.n, model.weights_entropy


outdf = pd.DataFrame(out.values())
outdf.index = out.keys()
outdf.columns = ["elbo", "df", "n", "entropy"]
outdf["bic"] = -2*outdf["elbo"] + outdf["df"] * np.log(outdf["n"])
outdf["aic"] = -2*outdf["elbo"] + 2*outdf["df"]
# outdf["gic"] = -2*outdf["elbo"] + outdf["df"] * np.log(np.log(outdf["n"]))
# outdf["gic2"] = -2*outdf["elbo"] + outdf["df"] * np.log(np.log(outdf["n"])) * np.log(outdf["n"])
outdf["elbo2"] = -2*outdf["elbo"] + outdf["entropy"] * 2
outdf["ebic1"] = -2*outdf["elbo"] + outdf["df"] * np.log(outdf["n"]) + np.log(outdf["n"] * outdf["df"])

print(outdf)


# import cProfile
# cProfile.run("model.fit(max_iter=100)")


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
