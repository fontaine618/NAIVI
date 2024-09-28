from NAIVI.initialization.usvt import initialize_latent_variables
from NAIVI.mle import GLM
from NAIVI.utils.data import JointDataset
import torch


def initialize(train: JointDataset, K: int, mnar: bool = False,
               estimate_variance: bool = False, verbose: bool = True,
               estimate_components: bool = False):
	out = dict()
	out["positions"] = dict()
	out["heterogeneity"] = dict()
	p_bin = train.p_bin_no_missingness
	p_cts = train.p_cts
	N = train.N
	i0, i1, A, j, X_cts, X_bin = train[:]
	# latent mean
	if verbose:
		print("Initializing latent variables")
	alpha_mean, Z_mean, components = initialize_latent_variables(i0, i1, A, K, estimate_components=estimate_components)
	out["positions"]["mean"] = Z_mean
	out["heterogeneity"]["mean"] = alpha_mean
	out["components"] = components
	if verbose:
		print("Initializing model parameters")
	glm = GLM(K, N, p_cts, p_bin, mnar=mnar, latent_positions=Z_mean)
	glm.fit(train, None, eps=1.e-6, max_iter=200, lr=0.1)
	with torch.no_grad():
		out["bias"] = glm.model.mean_model.bias.data
		out["weight"] = glm.model.mean_model.weight.data.t()
		out["sig2"] = glm.model.cts_logvar.data.exp()
	return out