import stan
import numpy as np
import torch
from .stan_models import model_cts, model_bin, model_both, model_none


class MCMC:

	def __init__(self, K, N, p_cts, p_bin,
				 position_prior=(0., 1.),
				 heterogeneity_prior=(-2., 1.)):
		self.p = int(p_cts + p_bin)
		self.p_bin = int(p_bin)
		self.p_cts = int(p_cts)
		self.K = int(K)
		self.N = int(N)
		self.heterogeneity_prior = heterogeneity_prior
		self.position_prior = position_prior
		# find which stan model to use
		if p_bin > 0 and p_cts > 0:
			self.model = "both"
			self.stan_model = model_both
		elif p_bin > 0 and p_cts == 0:
			self.model = "bin"
			self.stan_model = model_bin
		elif p_bin == 0 and p_cts > 0:
			self.model = "cts"
			self.stan_model = model_cts
		else:
			self.model = "none"
			self.stan_model = model_none
		self._init = [{"sig2_X": 1. * np.ones(self.p_cts)}]
		self._fit = None
		self._model = None

	def init(self, positions=None, heterogeneity=None, bias=None, weight=None):
		with torch.no_grad():
			if positions is not None:
				self._init[0]["Z"] = positions.detach().numpy()
			if heterogeneity is not None:
				self._init[0]["alpha"] = heterogeneity.flatten().detach().numpy()
			if bias is not None:
				self._init[0]["B"] = weight.detach().numpy()
			if weight is not None:
				self._init[0]["B0"] = bias.detach().numpy()

	def fit(self, train, test=None, Z_true=None, reg=0.,
			batch_size=100, eps=1.0e-6, max_iter=1000,
			lr=0.001, weight_decay=0., verbose=True,
			alpha_true=None, power=0.):
		# get data
		with torch.no_grad():
			i0, i1, A, j, X_cts, X_bin = train[:]
			E = i0.shape[0]
			if X_cts is not None:
				X_cts = X_cts.detach().numpy()
			if X_bin is not None:
				X_bin = X_bin.detach().int().numpy()
			A = A.detach().int().numpy().flatten()
			i0 = i0.detach().int().numpy() + 1
			i1 = i1.detach().int().numpy() + 1
			data = {
				"N": self.N, "E": E, "A": A, "i0": i0, "i1": i1,
				"K": self.K,
				"mu_alpha": self.heterogeneity_prior[0], "sig2_alpha": self.heterogeneity_prior[1],
				"mu_Z": self.position_prior[0], "sig2_Z": self.position_prior[1]
			}
			if self.model == "cts":
				data.update({
					"p": self.p, "p_cts": self.p_cts,
					"X_cts": X_cts,
				})
			elif self.model == "bin":
				data.update({
					"p": self.p, "p_bin": self.p_bin,
					"X_bin": X_bin
				})
			elif self.model == "both":
				data.update({
					"p": self.p, "p_cts": self.p_cts, "p_bin": self.p_bin,
					"X_cts": X_cts, "X_bin": X_bin
				})
			else:
				pass
		self._model = stan.build(self.stan_model, data=data, random_seed=0)
		self._fit = self._model.sample(
			num_chains=1, num_warmup=max_iter,
			num_samples=max_iter, init=self._init
		)

	def get(self, x):
		return self._fit.get(x)

	def posterior_mean(self, x):
		return self.get(x).mean(-1)
