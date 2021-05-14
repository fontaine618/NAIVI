from NAIVI.initialization.usvt import initialize_latent_variables
from NAIVI.mle import GLM
from NAIVI.advi import ADVI
from NAIVI.utils.data import JointDataset
import torch
import copy


def initialize(train: JointDataset, K: int, mnar: bool = False,
               estimate_variance: bool = False, verbose: bool = True):
	p_bin = train.p_bin_no_missingness
	p_cts = train.p_cts
	N = train.N
	i0, i1, A, j, X_cts, X_bin = train[:]
	# latent mean
	if verbose:
		print("Initializing latent variables")
	alpha_mean, Z_mean = initialize_latent_variables(i0, i1, A, K)
	# improve variance
	if estimate_variance:
		if verbose:
			print("Updating initial variances")
		train_no_cov = JointDataset(i0, i1, A)
		model = ADVI(K, N, 0, 0, mnar=mnar)
		model.init(positions=Z_mean, heterogeneity=alpha_mean)
		model.fit(train_no_cov, None, batch_size=len(train), eps=1.e-6, max_iter=100, lr=1.)
		Z_mean = model.model.encoder.latent_position_encoder.mean_encoder.values
		Z_log_var = model.model.encoder.latent_position_encoder.log_var_encoder.values
		alpha_mean = model.model.encoder.latent_heterogeneity_encoder.mean_encoder.values
		alpha_log_var = model.model.encoder.latent_heterogeneity_encoder.log_var_encoder.values
	else:
		Z_log_var = None
		alpha_log_var = None
	if verbose:
		print("Initializing model parameters")
	glm = GLM(K, N, p_cts, p_bin, mnar=mnar, latent_positions=Z_mean)
	glm.fit(train, None, batch_size=len(train), eps=1.e-6, max_iter=100, lr=1.)
	with torch.no_grad():
		B0 = glm.model.mean_model.bias.data
		B = glm.model.mean_model.weight.data
		log_var = glm.model.cts_logvar.data
	return (alpha_mean, alpha_log_var), (Z_mean, Z_log_var), B0, B, log_var