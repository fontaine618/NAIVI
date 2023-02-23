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


N = 50
p_bin = 10
p_cts = 10

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
	latent_dim=2,
	heterogeneity_prior_mean=-1.5
)
model.fit(max_iter=100, rel_tol=1e-6)


self = model.factors["affine_cts"]


self.fit_and_evaluate(
	max_iter=100,
	true_values=true_values
)

self.evaluate(true_values)


import cProfile
cProfile.run("vmp.fit_and_evaluate(max_iter=100, true_values=true_values)")






# plot factor graphs
import networkx as nx
factors = self.factors
variables = self.variables
messages = self.messages

G = nx.DiGraph()
for factor in factors.values():
	G.add_node(repr(factor))
	G.nodes[repr(factor)]["shape"] = "s"
for variable in variables.values():
	G.add_node(repr(variable))
for message in messages:
	G.add_edge(repr(message.factor), repr(message.variable), label=repr(message))


plt.figure(figsize=(20, 20))
layout = nx.planar_layout(G)
nx.draw_networkx_nodes(G, node_shape="s", nodelist=[repr(factor) for factor in factors.values()],
					   pos=layout, node_color="k")
nx.draw_networkx_nodes(G, node_shape="o", nodelist=[repr(variable) for variable in variables.values()],
					   pos=layout, node_color="w", edgecolors="k")
nx.draw_networkx_edges(G, pos=layout, arrows=False)
nx.draw_networkx_edge_labels(G, pos=layout, label_pos=0.5, edge_labels=nx.get_edge_attributes(G, "label"))
plt.show()







# ELBO History plot
df = pd.DataFrame(vmp.elbo_history)
df = df.loc[:, df.var() > 0.]
df.plot()
plt.xscale("log")
plt.title("ELBOs")
plt.show()

# ELBO MC History plot
df = pd.DataFrame(vmp.elbo_mc_history)
df = df.loc[:, df.var() > 0.]
df.plot()
plt.xscale("log")
plt.title("ELBOs (MC)")
plt.show()

# Metric plot
df = pd.DataFrame(vmp.metrics_history["X_bin_missing_auroc"])
df = df.loc[:, df.var() > 0.]
df.plot()
plt.xscale("log")
plt.show()





# For experiment

N = 200
K = 3
p_cts = 50
p_bin = 50
p = p_cts + p_bin
var_cov = 1.
missing_mean = 0.
seed = 0
alpha_mean_gen = -1.85
mnar_sparsity = 0.
adjacency_noise = 0.
constant_components = True
K_model = 3
mnar = False
alpha_mean_model = 0.
reg = 0.
network_weight = 1.
estimate_components = False
algo = "VMP"
max_iter = 100
n_samples = 0
eps = 1e-5
keep_logs = True
power = 1.
init_method = ""
optimizer = "RProp"
lr = 0.01
mcmc_n_sample = 0
mcmc_n_chains = 0
mcmc_n_warmup = 0
mcmc_n_thin = 0





# devel _evaluate
name = "X_cts"
value = A
self = vmp


vmp.factors["affine_cts"].parameters["weights"].data = B[:, :p_cts]
vmp.factors["affine_bin"].parameters["weights"].data = B[:, p_cts:]
vmp.factors["affine_cts"].parameters["bias"].data = B0[0, :p_cts]
vmp.factors["affine_bin"].parameters["bias"].data = B0[0, p_cts:]

disable_logging()

print([x for x in vmp._elbo().values() if x != 0])
print(vmp.elbo().item())
for iter in range(25):
	with torch.no_grad():
		vmp._e_step()
		vmp._m_step()
		print([x for x in vmp._elbo().values() if x != 0])
		print(iter, vmp.elbo().item())

self = vmp.factors["affine_cts"]


vmp.variables["latent"].posterior.mean[0:10, :]
Z[0:10, :]


# test update of variance
self = vmp.factors["cts_model"]

for i in range(1000):
	elbo = self.elbo()
	elbo.backward()
	self.hyperparameters["log_variance"].data += 0.00001 * self.hyperparameters["log_variance"].grad
	print(i, elbo.item())

print(self.hyperparameters["log_variance"].data.exp())
self.update_parameters()
print(self.elbo().item())


# test update affine parameters

self = vmp.factors["affine_cts"]
self.update_parameters()
self.hyperparameters["weights"].data
B[:, :10]
self.hyperparameters["bias"].data
B0[0, :10]
vmp.factors["cts_model"].hyperparameters["log_variance"].data.exp()


# Logistic ELBO

self = vmp.factors["bin_model"]

# Samples and forward

with torch.no_grad():
	vmp._elbo()
	vmp.sample(100)
	vmp.forward()
	vmp._elbo_mc()
	vmp.sample(100)
	vmp._elbo_mc()

self = vmp.factors["cts_model"]

self = MultivariateNormal.from_mean_and_variance(
	mean=torch.ones((2, 3)),
	variance=torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
)

self.sample(7)

self = vmp.variables["latent"].posterior



# {'latent': [v0] Variable,
#  'heterogeneity': [v1] Variable,
#  'mean_bin': [v2] Variable,
#  'mean_cts': [v3] Variable,
#  'bin_obs': [v4] Variable,
#  'cts_obs': [v5] Variable,
#  'left_heterogeneity': [v8] Variable,
#  'right_heterogeneity': [v9] Variable,
#  'left_latent': [v10] Variable,
#  'right_latent': [v11] Variable,
#  'inner_product': [v12] Variable,
#  'edge_probit': [v13] Variable,
#  'edge': [v14] Variable}


# {'latent_prior': [f0] MultivariateNormalPrior,
#  'heterogeneity_prior': [f1] NormalPrior,
#  'affine_bin': [f2] Affine,
#  'affine_cts': [f3] Affine,
#  'bin_model': [f4] Logistic,
#  'cts_model': [f5] GaussianFactor,
#  'bin_observed': [f6] ObservedFactor,
#  'cts_observed': [f7] ObservedFactor,
#  'select_left_heterogeneity': [f8] Select,
#  'select_right_heterogeneity': [f9] Select,
#  'select_left_latent': [f10] Select,
#  'select_right_latent': [f11] Select,
#  'inner_product_factor': [f12] InnerProduct,
#  'edge_sum': [f13] Sum,
#  'edge_model': [f14] Logistic,
#  'edge_observed': [f15] ObservedFactor}


# # VMP
# train      loss             -37021.163457
# train      mse               nan
# train      auc              0.876003
# train      auc_A            0.911218
# error      ZZt              0.079061
# error      Theta_X          0.340332
# error      Theta_A          0.030570
# error      P                0.002404
# error      BBt              0.537513
# error      alpha            0.101361
# test       mse               nan
# test       auc              0.854116
# test       auc_A            0.911218
# train      time             120.855487
# data       density          0.096096
# data       missing_prop     0.268500
#
# # MLE
# train      grad_Linfty      0.000003
# train      grad_L1          0.000280
# train      grad_L2          0.000009
# train      loss             39273.627060
# train      mse              0.000000
# train      auc              0.886348
# train      auc_A            0.919652
# test       loss             30270.588934
# test       mse              0.000000
# test       auc              0.855271
# test       auc_A            0.919652
# error      ZZt              0.067644
# error      P                0.002274
# error      Theta_X          0.108714
# error      Theta_A          0.025730
# error      BBt              0.294152
# error      alpha            0.084860
# train      time             1.570883
# data       density          0.096096
# data       missing_prop     0.268500
#
# # ADVI
# train      grad_Linfty      0.000003
# train      grad_L1          0.000353
# train      grad_L2          0.000009
# train      loss             55946.405569
# train      mse              0.000000
# train      auc              0.875910
# train      auc_A            0.920160
# test       loss             58360.800399
# test       mse              0.000000
# test       auc              0.849639
# test       auc_A            0.920160
# error      ZZt              0.080062
# error      P                0.002452
# error      Theta_X          0.339360
# error      Theta_A          0.030735
# error      BBt              0.436466
# error      alpha            0.103582
# train      time             2.847495
# data       density          0.096096
# data       missing_prop     0.268500


n = 3
H = torch.eye(n) * n - 1
torch.linalg.slogdet(H)
torch.linalg.eigvalsh(H)
torch.linalg.eigh(H)