import torch
import math
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from NAIVI.vmp import disable_logging
from NAIVI.vmp.vmp import VMP
from NAIVI_experiments.gen_data_mnar import generate_dataset

# for debugging
from NAIVI.vmp.distributions.normal import MultivariateNormal, Normal, _batch_mahalanobis
from NAIVI.vmp.factors.factor import Factor
from NAIVI.vmp.variables.variable import Variable
from NAIVI.vmp.messages.message import Message

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
		missing_mean=0.5,
		alpha_mean=-1.5,
		seed=0,
		mnar_sparsity=0.,
		adjacency_noise=0.,
		constant_components=True
	)

vmp = VMP(
	n_nodes=N,
	binary_covariates=X_bin,
	continuous_covariates=X_cts,
	edges=A,
	edge_index_left=i0,
	edge_index_right=i1,
	latent_dim=3
)

vmp.factors["affine_cts"].parameters["weights"].data = B[:, :p_cts]
vmp.factors["affine_bin"].parameters["weights"].data = B[:, p_cts:]
vmp.factors["affine_cts"].parameters["bias"].data = B0[0, :p_cts]
vmp.factors["affine_bin"].parameters["bias"].data = B0[0, p_cts:]

disable_logging()

print([x for x in vmp._elbo().values() if x != 0])
print(vmp.elbo().item())
for iter in range(25):
	print(iter)
	with torch.no_grad():
		vmp._e_step()
		# vmp._m_step()
		print([x for x in vmp._elbo().values() if x != 0])
		print(vmp.elbo().item())

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