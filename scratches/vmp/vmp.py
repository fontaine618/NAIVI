import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from NAIVI.vmp import disable_logging
from NAIVI.vmp.vmp import VMP
from NAIVI_experiments.gen_data_mnar import generate_dataset

# for debugging
from NAIVI.vmp.distributions.normal import MultivariateNormal, Normal, _batch_mahalanobis
from NAIVI.vmp.factors.factor import Factor
from NAIVI.vmp.variables.variable import Variable
from NAIVI.vmp.messages.message import Message

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, \
	i0, i1, A, B, B0, C, C0, W = \
	generate_dataset(
		N=100,
		K=3,
		p_cts=10,
		p_bin=10,
		var_cov=1.,
		missing_mean=0.5,
		alpha_mean=-1.5,
		seed=0,
		mnar_sparsity=0.,
		adjacency_noise=0.,
		constant_components=True
	)

vmp = VMP(
	binary_covariates=X_bin,
	continuous_covariates=X_cts,
	edges=A,
	edge_index_left=i0,
	edge_index_right=i1,
	latent_dim=3
)

vmp._factors["affine_cts"].parameters["weights"].data = B[:, :10]
vmp._factors["affine_bin"].parameters["weights"].data = B[:, 10:]
vmp._factors["affine_cts"].parameters["bias"].data = B0[0, :10]
vmp._factors["affine_bin"].parameters["bias"].data = B0[0, 10:]

for iter in range(10):
	print(iter)
	with torch.no_grad():
		vmp._e_step()

self = vmp._factors["affine_cts"]


vmp._variables["latent"].posterior.mean[0:10, :]
Z[0:10, :]


# test update of variance
self = vmp._factors["cts_model"]

for i in range(1000):
	elbo = self.elbo()
	elbo.backward()
	self.parameters["log_variance"].data += 0.00001 * self.parameters["log_variance"].grad
	print(i, elbo.item())

print(self.parameters["log_variance"].data.exp())
self.update_parameters()
print(self.elbo().item())


# test update affine parameters

self = vmp._factors["affine_cts"]
vmp._factors["cts_model"].elbo()
self.update_parameters()
vmp._factors["cts_model"].elbo()
self.parameters["weights"].data
B[:, :10]
self.parameters["bias"].data
B0[0, :10]

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