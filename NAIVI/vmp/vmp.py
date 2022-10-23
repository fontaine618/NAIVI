from __future__ import annotations
import itertools
import torch
import numpy as np
from typing import Tuple, Dict

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

	_default_parameters = {
		"latent_prior_variance": 1.,
		"latent_prior_mean": 0.,
		"heterogeneity_prior_variance": 1.,
		"heterogeneity_prior_mean": 0.
	}

	"""Forward sequence, from latent to observed."""
	_vmp_sequence = [
		"latent_prior",
		"heterogeneity_prior",
		"affine_bin",
		"affine_cts",
		"bin_model",
		"cts_model",
		"bin_observed",
		"cts_observed",
		"select_left_heterogeneity",
		"select_right_heterogeneity",
		"select_left_latent",
		"select_right_latent",
		"inner_product_factor",
		"edge_sum",
		"edge_model",
		"edge_observed"
	]

	def __init__(
		self,
		binary_covariates: torch.Tensor,
		continuous_covariates: torch.Tensor,
		edges: torch.Tensor,
		edge_index_left: torch.Tensor,
		edge_index_right: torch.Tensor,
		latent_dim: int,
		**kwargs
	):
		self._prepare_parameters(**kwargs)
		self._initialize_model(
			binary_covariates,
			continuous_covariates,
			edge_index_left,
			edge_index_right,
			edges,
			latent_dim
		)

	def _prepare_parameters(self, **kwargs):
		self._parameters = self._default_parameters.copy()
		self._parameters.update(kwargs)

	def _initialize_model(
		self,
		binary_covariates,
		continuous_covariates,
		edge_index_left,
		edge_index_right,
		edges,
		latent_dim
	):
		# dimensions
		N, p_bin = binary_covariates.shape
		p_cts = continuous_covariates.shape[1]
		K = latent_dim
		ne = edges.shape[0]
		# initialize factors and variables
		latent_prior = MultivariateNormalPrior(
			dim=K,
			mean=self._parameters["latent_prior_mean"],
			variance=self._parameters["latent_prior_variance"]
		)
		latent = Variable((N, K))
		heterogeneity_prior = NormalPrior(
			mean=self._parameters["heterogeneity_prior_mean"],
			variance=self._parameters["heterogeneity_prior_variance"]
		)
		heterogeneity = Variable((N, 1))
		# covariate branch
		affine_bin = Affine(K, p_bin, latent)
		affine_cts = Affine(K, p_cts, latent)
		mean_bin = Variable((N, p_bin))
		mean_cts = Variable((N, p_cts))
		bin_model = Logistic(p_bin, parent=mean_bin, method="quadratic")
		cts_model = GaussianFactor(p_cts, parent=mean_cts)
		bin_obs = Variable((N, p_bin))
		cts_obs = Variable((N, p_cts))
		bin_observed = ObservedFactor(binary_covariates, parent=bin_obs)
		cts_observed = ObservedFactor(continuous_covariates, parent=cts_obs)
		# edge branch
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
		latent_prior.set_children(child=latent)
		heterogeneity_prior.set_children(child=heterogeneity)
		affine_bin.set_children(child=mean_bin)
		affine_cts.set_children(child=mean_cts)
		bin_model.set_children(child=bin_obs)
		cts_model.set_children(child=cts_obs)
		select_left_heterogeneity.set_children(child=left_heterogeneity)
		select_right_heterogeneity.set_children(child=right_heterogeneity)
		select_left_latent.set_children(child=left_latent)
		select_right_latent.set_children(child=right_latent)
		inner_product_factor.set_children(child=inner_product)
		edge_sum.set_children(child=edge_probit)
		edge_model.set_children(child=edge)
		# store
		self._factors = {
			"latent_prior": latent_prior,
			"heterogeneity_prior": heterogeneity_prior,
			"affine_bin": affine_bin,
			"affine_cts": affine_cts,
			"bin_model": bin_model,
			"cts_model": cts_model,
			"bin_observed": bin_observed,
			"cts_observed": cts_observed,
			"select_left_heterogeneity": select_left_heterogeneity,
			"select_right_heterogeneity": select_right_heterogeneity,
			"select_left_latent": select_left_latent,
			"select_right_latent": select_right_latent,
			"inner_product_factor": inner_product_factor,
			"edge_sum": edge_sum,
			"edge_model": edge_model,
			"edge_observed": edge_observed
		}
		self._variables = {
			"latent": latent,
			"heterogeneity": heterogeneity,
			"mean_bin": mean_bin,
			"mean_cts": mean_cts,
			"bin_obs": bin_obs,
			"cts_obs": cts_obs,
			"left_heterogeneity": left_heterogeneity,
			"right_heterogeneity": right_heterogeneity,
			"left_latent": left_latent,
			"right_latent": right_latent,
			"inner_product": inner_product,
			"edge_probit": edge_probit,
			"edge": edge
		}
		self._initialize_posterior()
		self._vmp_forward()

	def _initialize_posterior(self):
		for variable in self._variables.values():
			variable.compute_posterior()

	def _vmp_backward(self):
		for fname in self._vmp_sequence[::-1]:
			self._factors[fname].update_messages_from_children()
			self._factors[fname].update_messages_to_parents()

	def _vmp_forward(self):
		for fname in self._vmp_sequence:
			self._factors[fname].update_messages_from_parents()
			self._factors[fname].update_messages_to_children()

	def _e_step(self, n_iter: int = 1):
		for _ in range(n_iter):
			self._vmp_backward()
			self._vmp_forward()

