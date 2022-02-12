import stan
import numpy as np
import torch
import arviz as az
import pandas as pd
from NAIVI.mcmc.stan_models import model_cts, model_bin, model_both, model_none
from NAIVI import NAIVI


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
		self._init = [{"sig2_X": 1. * np.ones(self.p_cts)}]*10
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

	def fit(self, train, n_sample=1000, num_chains=1, num_warmup=1000,
	        verbose=True, true_values=None, **kwargs):
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
			num_chains=num_chains, num_warmup=num_warmup,
			num_samples=n_sample, init=[self._init[0] for _ in range(num_chains)]
		)
		return self.metrics(train, true_values)

	def get(self, x):
		return self._fit.get(x)

	def posterior_mean(self, x):
		return self.get(x).mean(-1)

	def diagnostics(self, var):
		return az.summary(self._fit, var, kind="diagnostics", round_to=5)

	def diagnostic_summary(self):
		ZZt_diag = self.diagnostics("ZZt").describe().transpose()
		ZZt_diag.index = pd.MultiIndex.from_product([["ZZt"], ZZt_diag.index])
		alpha_diag = self.diagnostics("alpha").describe().transpose()
		alpha_diag.index = pd.MultiIndex.from_product([["alpha"], alpha_diag.index])
		Theta_A_diag = self.diagnostics("Theta_A").describe().transpose()
		Theta_A_diag.index = pd.MultiIndex.from_product([["Theta_A"], Theta_A_diag.index])
		if self.p > 0:
			B0_diag = self.diagnostics("B0").describe().transpose()
			B0_diag.index = pd.MultiIndex.from_product([["B0"], B0_diag.index])
			Theta_X_diag = self.diagnostics("Theta_X").describe().transpose()
			Theta_X_diag.index = pd.MultiIndex.from_product([["Theta_X"], Theta_X_diag.index])
		else:
			B0_diag = None
			Theta_X_diag = None
		diagnostics = pd.concat([ZZt_diag, B0_diag, alpha_diag, Theta_X_diag, Theta_A_diag])
		diagnostics = diagnostics.melt(ignore_index=False).reset_index().set_index(
			["level_0", "level_1", "variable"]).transpose()
		return diagnostics

	def metrics(self, train, true_values):
		out = dict()
		out[("train", "mse")], \
		out[("train", "auc")], \
		out[("train", "auc_A")] = self.evaluate(train)
		# estimation error
		for k, v in true_values.items():
			try:
				v = v.detach().cpu().numpy()
				if k == "ZZt":
					ZZt = self.posterior_mean("ZZt")
					value = ((ZZt - v) ** 2).sum() / (v ** 2).sum()
				elif k == "Theta_X" and self.p > 0:
					Theta_X = self.posterior_mean("Theta_X")
					value = ((Theta_X - v) ** 2).mean()
				elif k == "Theta_A":
					Theta_A = self.posterior_mean("Theta_A")
					value = ((Theta_A - v) ** 2).mean()
				elif k == "P":
					P = self.posterior_mean("proba")
					value = ((P - v) ** 2).mean()
				elif k == "BBt" and self.p > 0:
					BBt = self.posterior_mean("BBt")
					value = ((BBt - v) ** 2).sum() / (v ** 2).sum()
				elif k == "alpha":
					alpha = self.posterior_mean("alpha")
					value = ((alpha - v) ** 2).mean()
				else:
					value = np.nan
			except Exception:
				value = np.nan
			out[("error", k)] = value
		return out

	def evaluate(self, train):
		with torch.no_grad():
			_, _, A, _, X_cts, X_bin = train[:]
			# train mse, auc, auc_A
			if self.p > 0:
				Theta_X = self.posterior_mean("Theta_X")
				Theta_X_split = np.hsplit(Theta_X, [self.p_cts, self.p_cts + self.p_bin])
				mean_cts = torch.Tensor(Theta_X_split[0])
				proba_bin = torch.Tensor(1. / (1. + np.exp(-Theta_X_split[1])))
			else:
				mean_cts = None
				proba_bin = None
			proba_adj = torch.Tensor(self.posterior_mean("proba"))
			auc, mse, auc_A = NAIVI.prediction_metrics(
				X_bin=X_bin, X_cts=X_cts, A=A,
				mean_cts=mean_cts,
				proba_bin=proba_bin,
				proba_adj=proba_adj
			)
		return auc, mse, auc_A


