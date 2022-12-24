from __future__ import annotations
import itertools
import torch
import numpy as np
from typing import Tuple, Dict, Optional

from .factors.affine import Affine
from .factors.gaussian import GaussianFactor
from .factors.normal_prior import MultivariateNormalPrior, NormalPrior
from .factors.observedfactor import ObservedFactor
from .factors.logistic import Logistic
from .factors.select import Select
from .factors.sum import Sum
from .factors.factor import Factor
from .factors.inner_product import InnerProduct

from .variables.variable import Variable
from .messages.message import Message
from .distributions.normal import MultivariateNormal, Normal


class VMP:
	"""
	NAIVI using Variational message passing with multivariate posterior.
	"""

	_default_hyperparameters = {
		"latent_prior_variance": 1.,
		"latent_prior_mean": 0.,
		"heterogeneity_prior_variance": 1.,
		"heterogeneity_prior_mean": 0.
	}

	def __init__(
		self,
		n_nodes: int,
		latent_dim: int,
		binary_covariates: torch.Tensor | None,
		continuous_covariates: torch.Tensor | None,
		edges: torch.Tensor | None,
		edge_index_left: torch.Tensor | None,
		edge_index_right: torch.Tensor | None,
		**kwargs
	):
		self._prepare_hyperparameters(**kwargs)
		self.latent_dim = latent_dim
		self.factors = dict()
		self.variables = dict()
		self._vmp_sequence = list()
		self._m_step_factors = list()
		self._initialize_model(
			n_nodes=n_nodes,
			latent_dim=latent_dim,
			binary_covariates=binary_covariates,
			continuous_covariates=continuous_covariates,
			edge_index_left=edge_index_left,
			edge_index_right=edge_index_right,
			edges=edges,
		)
		self.elbo_history = dict()
		self.metrics_history = dict()

	def _prepare_hyperparameters(self, **kwargs):
		self.hyperparameters = self._default_hyperparameters.copy()
		self.hyperparameters.update(kwargs)

	def _initialize_model(
		self,
		n_nodes,
		latent_dim,
		binary_covariates,
		continuous_covariates,
		edge_index_left,
		edge_index_right,
		edges,
	):
		N = n_nodes
		K = latent_dim

		# initialize factors and variables
		heterogeneity, latent = self._initialize_priors(K, N)
		self._initialize_binary_model(K, N, binary_covariates, latent)
		self._initialize_continuous_model(K, N, continuous_covariates, latent)
		self._initialize_edge_model(K, edge_index_left, edge_index_right, edges, heterogeneity, latent)
		self._break_symmetry()
		self._initialize_posterior()
		self._vmp_forward()

	def _initialize_edge_model(self, K, edge_index_left, edge_index_right, edges, heterogeneity, latent):
		if (edges is None) or (edge_index_right is None) or (edge_index_left is None):
			return
		ne = edges.shape[0]
		if ne == 0:
			return
		select_left_heterogeneity = Select(edge_index_left, heterogeneity, Normal)
		select_right_heterogeneity = Select(edge_index_right, heterogeneity, Normal)
		left_heterogeneity = Variable((ne, 1))
		right_heterogeneity = Variable((ne, 1))
		select_left_latent = Select(edge_index_left, latent, MultivariateNormal)
		select_right_latent = Select(edge_index_right, latent, MultivariateNormal)
		left_latent = Variable((ne, K))
		right_latent = Variable((ne, K))
		inner_product_factor = InnerProduct(left_latent, right_latent)
		inner_product = Variable((ne, 1))
		edge_sum = Sum(
			inner_product=inner_product,
			left_heterogeneity=left_heterogeneity,
			right_heterogeneity=right_heterogeneity
		)
		edge_probit = Variable((ne, 1))
		edge_model = Logistic(1, parent=edge_probit, method="quadratic")
		edge = Variable((ne, 1))
		edge_observed = ObservedFactor(edges, parent=edge)
		# attach children
		select_left_heterogeneity.set_children(child=left_heterogeneity)
		select_right_heterogeneity.set_children(child=right_heterogeneity)
		select_left_latent.set_children(child=left_latent)
		select_right_latent.set_children(child=right_latent)
		inner_product_factor.set_children(child=inner_product)
		edge_sum.set_children(child=edge_probit)
		edge_model.set_children(child=edge)
		self.factors.update({
			"select_left_heterogeneity": select_left_heterogeneity,
			"select_right_heterogeneity": select_right_heterogeneity,
			"select_left_latent": select_left_latent,
			"select_right_latent": select_right_latent,
			"inner_product_factor": inner_product_factor,
			"edge_sum": edge_sum,
			"edge_model": edge_model,
			"edge_observed": edge_observed,
		})
		self.variables.update({
			"left_heterogeneity": left_heterogeneity,
			"right_heterogeneity": right_heterogeneity,
			"left_latent": left_latent,
			"right_latent": right_latent,
			"inner_product": inner_product,
			"edge_probit": edge_probit,
			"edge": edge,
		})
		self._vmp_sequence.extend([
			"select_left_heterogeneity",
			"select_right_heterogeneity",
			"select_left_latent",
			"select_right_latent",
			"inner_product_factor",
			"edge_sum",
			"edge_model",
			"edge_observed"
		])
		self._m_step_factors.extend([])

	def _initialize_continuous_model(self, K, N, continuous_covariates, latent):
		if continuous_covariates is None:
			return
		p_cts = continuous_covariates.shape[1]
		if p_cts == 0:
			return
		affine_cts = Affine(K, p_cts, latent)
		mean_cts = Variable((N, p_cts))
		cts_model = GaussianFactor(p_cts, parent=mean_cts)
		cts_obs = Variable((N, p_cts))
		cts_observed = ObservedFactor(continuous_covariates, parent=cts_obs)
		affine_cts.set_children(child=mean_cts)
		cts_model.set_children(child=cts_obs)
		self.factors.update({
			"affine_cts": affine_cts,
			"cts_model": cts_model,
			"cts_observed": cts_observed,
		})
		self.variables.update({
			"mean_cts": mean_cts,
			"cts_obs": cts_obs,
		})
		self._vmp_sequence.extend([
			"affine_cts",
			"cts_model",
			"cts_observed"
		])
		self._m_step_factors.extend([
			"affine_cts",
			"cts_model",
		])

	def _initialize_binary_model(self, K, N, binary_covariates, latent):
		if binary_covariates is None:
			return
		p_bin = binary_covariates.shape[1]
		if p_bin == 0:
			return
		affine_bin = Affine(K, p_bin, latent)
		logit_bin = Variable((N, p_bin))
		bin_model = Logistic(p_bin, parent=logit_bin, method="quadratic")
		bin_obs = Variable((N, p_bin))
		bin_observed = ObservedFactor(binary_covariates, parent=bin_obs)
		affine_bin.set_children(child=logit_bin)
		bin_model.set_children(child=bin_obs)
		self.factors.update({
			"affine_bin": affine_bin,
			"bin_model": bin_model,
			"bin_observed": bin_observed,
		})
		self.variables.update({
			"logit_bin": logit_bin,
			"bin_obs": bin_obs,
		})
		self._vmp_sequence.extend([
			"affine_bin",
			"bin_model",
			"bin_observed"
		])
		self._m_step_factors.extend([
			"affine_bin",
		])

	def _initialize_priors(self, K, N):
		latent_prior = MultivariateNormalPrior(
			dim=K,
			mean=self.hyperparameters["latent_prior_mean"],
			variance=self.hyperparameters["latent_prior_variance"]
		)
		latent = Variable((N, K))
		heterogeneity_prior = NormalPrior(
			mean=self.hyperparameters["heterogeneity_prior_mean"],
			variance=self.hyperparameters["heterogeneity_prior_variance"]
		)
		heterogeneity = Variable((N, 1))
		latent_prior.set_children(child=latent)
		heterogeneity_prior.set_children(child=heterogeneity)
		self.factors.update({
			"latent_prior": latent_prior,
			"heterogeneity_prior": heterogeneity_prior,
		})
		self.variables.update({
			"latent": latent,
			"heterogeneity": heterogeneity,
		})
		self._vmp_sequence.extend([
			"latent_prior",
			"heterogeneity_prior"
		])
		self._m_step_factors.extend([])
		return heterogeneity, latent

	def _initialize_posterior(self):
		for variable in self.variables.values():
			variable.compute_posterior()

	def _break_symmetry(self):
		# TODO implement this
		pass

	def _vmp_backward(self):
		for fname in self._vmp_sequence[::-1]:
			self.factors[fname].update_messages_from_children()
			self.factors[fname].update_messages_to_parents()

	def _vmp_forward(self):
		for fname in self._vmp_sequence:
			self.factors[fname].update_messages_from_parents()
			self.factors[fname].update_messages_to_children()

	def _e_step(self, n_iter: int = 1):
		for _ in range(n_iter):
			self._vmp_backward()
			self._vmp_forward()

	def _m_step(self):
		for fname in self._m_step_factors:
			self.factors[fname].update_parameters()

	@property
	def parameters(self) -> Dict[str, Dict[str, torch.nn.Parameter]]:
		parms = dict()
		for fname, factor in self.factors.items():
			if hasattr(factor, "parameters"):
				parms[fname] = factor.parameters
		return parms

	def elbo(self) -> float:
		return sum([factor.elbo() for factor in self.factors.values()])

	def _elbo(self):
		return {
			fname: factor.elbo().item()
			for fname, factor in self.factors.items()
		}

	def elbo_mc(self) -> float:
		"""Compute approximate elbo using samples from the posterior.
		NB: the samples can be obtained in two ways:
		- sample latent variables + forward
		- sample all
		"""
		return sum([factor.elbo_mc() for factor in self.factors.values()])

	def _elbo_mc(self):
		return {
			fname: factor.elbo_mc()
			for fname, factor in self.factors.items()
		}

	def sample(self, n_samples: int = 1):
		for var in self.variables.values():
			var.sample(n_samples)

	def forward(self, n_samples: int = 1):
		# sample latent variables
		self.variables["latent"].sample(n_samples)
		self.variables["heterogeneity"].sample(n_samples)
		# propagate through model
		for fname in self._vmp_sequence:
			self.factors[fname].forward()

	def _update_elbo_history(self, elbo: dict[str, float]):
		for k, v in elbo.items():
			if k not in self.elbo_history:
				self.elbo_history[k] = [v]
			self.elbo_history[k].append(v)

	def _update_metrics_history(self, metrics: Dict[str, float]):
		for k, v in metrics.items():
			if k not in self.metrics_history:
				self.metrics_history[k] = [v]
			self.metrics_history[k].append(v)

	def fit_and_evaluate(
			self,
			max_iter: int = 1000,
			rel_tol: float = 1e-6,
			verbose: bool = True,
			true_values: Dict[str, torch.Tensor] = {},
	):
		elbo = self.elbo()
		for i in range(max_iter):
			self._e_step()
			self._m_step()
			new_elbo = self.elbo().item()
			elbos = self._elbo()
			elbos["sum"] = new_elbo
			self._update_elbo_history(elbos)
			self.evaluate(true_values)
			increased = new_elbo > elbo
			if verbose:
				print(f"[VMP] Iteration {i:<4} Elbo: {new_elbo:.4f} {'' if increased else '(decreased)'}")
			if abs(new_elbo - elbo) < rel_tol * abs(elbo):
				break
			elbo = new_elbo

	def fit(self, max_iter: int = 1000, rel_tol: float = 1e-6, verbose: bool = True):
		self.fit_and_evaluate(max_iter, rel_tol, verbose)

	def evaluate(self, true_values: Dict[str, torch.Tensor] = {}):
		# add observed data
		if "bin_observed" in self.factors:
			true_values["X_bin"] = self.factors["bin_observed"].values.values
		if "cts_observed" in self.factors:
			true_values["X_cts"] = self.factors["cts_observed"].values.values
		if "edge_observed" in self.factors:
			true_values["A"] = self.factors["edge_observed"].values.values
		for name, value in true_values.items():
			self._evaluate(name, value)

	@property
	def weights(self) -> torch.Tensor:
		weights = torch.zeros(self.latent_dim, 0)
		if "affine_cts" in self.parameters:
			weights = torch.cat([weights, self.parameters["affine_cts"]["weights"].data], dim=1)
		if "affine_bin" in self.parameters:
			weights = torch.cat([weights, self.parameters["affine_bin"]["weights"].data], dim=1)
		return weights

	@property
	def bias(self) -> torch.Tensor:
		bias = torch.zeros(0)
		if "affine_cts" in self.parameters:
			bias = torch.cat([bias, self.parameters["affine_cts"]["bias"].data], dim=0)
		if "affine_bin" in self.parameters:
			bias = torch.cat([bias, self.parameters["affine_bin"]["bias"].data], dim=0)
		return bias.reshape(1, -1)

	def _evaluate(self, name: str, value: torch.Tensor):
		metrics = dict()
		# compute metrics
		if name == "heterogeneity":
			post = self.variables["heterogeneity"].posterior.mean
			diff = (post - value).abs()
			metrics["heteregeneity_l2"] = (diff ** 2).sum().sqrt().item()
		elif name == "latent":
			post = self.variables["latent"].posterior.mean
			ZZt = post @ post.T
			ZtZinv = torch.linalg.inv(post.T @ post)
			Proj = post @ ZtZinv @ post.T

			ZZt0 = value @ value.T
			ZtZinv0 = torch.linalg.inv(value.T @ value)
			Proj0 = value @ ZtZinv0 @ value.T

			metrics["latent_ZZt_fro"] = (ZZt - ZZt0).pow(2.).sum().sqrt().item()
			metrics["latent_Proj_fro"] = (Proj - Proj0).pow(2.).sum().sqrt().item()
		elif name == "bias":
			bias = self.bias
			diff = (bias - value).abs()
			metrics["bias_l2"] = (diff ** 2).sum().sqrt().item()
		elif name == "weights":
			weights = self.weights.T
			ZZt = weights @ weights.T
			ZtZinv = torch.linalg.inv(weights.T @ weights)
			Proj = weights @ ZtZinv @ weights.T

			value = value.T
			ZZt0 = value @ value.T
			ZtZinv0 = torch.linalg.inv(value.T @ value)
			Proj0 = value @ ZtZinv0 @ value.T

			metrics["weights_BBt_fro"] = (ZZt - ZZt0).pow(2.).sum().sqrt().item()
			metrics["weights_Proj_fro"] = (Proj - Proj0).pow(2.).sum().sqrt().item()
		elif name == "cts_noise":
			# TODO
			pass
		elif name == "Theta_X":
			# TODO
			pass
		elif name == "Theta_A":
			# TODO
			pass
		elif name == "P":
			# TODO
			pass
		elif name == "X_cts":
			# TODO
			pass
		elif name == "X_bin":
			# TODO
			pass
		elif name == "X_cts_missing":
			# TODO
			pass
		elif name == "X_bin_missing":
			# TODO
			pass
		elif name == "A":
			# TODO
			pass
		else:
			# could print a warning message, but that would appear every iteration ...
			pass
		# update history
		self._update_metrics_history(metrics)