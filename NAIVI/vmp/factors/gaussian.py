from __future__ import annotations
import torch
import math
from .factor import Factor
from ..distributions.normal import Normal, MultivariateNormal
from ..messages import Message

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class GaussianFactor(Factor):
	"""
	x ~ N(mu, s^2)

	Batches over all but the last dimension, which takes different noise variance.
	"""

	_name = "GaussianFactor"

	def __init__(self, dim: int, parent: Variable):
		super(GaussianFactor, self).__init__(parent=parent)
		self.parameters["log_variance"] = torch.nn.Parameter(torch.zeros(dim))

	def initialize_messages_to_parents(self):
		i = self._name_to_id["parent"]
		self.messages_to_parents[i] = GaussianFactorToParentMessage(self.parents[i], self)

	def initialize_messages_to_children(self):
		i = self._name_to_id["child"]
		self.messages_to_children[i] = GaussianFactorToChildMessage(self.children[i], self)

	def update_messages_to_children(self):
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfp = self.messages_to_parents[p].message_to_factor # N x p
		# mfp = self.parents[p].posterior # N x p
		m, v = mfp.mean_and_variance # N x p, N x p
		var = self.parameters["log_variance"].data.exp() # p
		v = v + var.unsqueeze(0).expand(m.shape[0], -1)
		self.messages_to_children[c].message_to_variable = Normal.from_mean_and_variance(m, v)

	def update_messages_to_parents(self):
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		# this is a bit hacky, but for some reason the message sent is just slitghly wrong
		# in such a way that we don't find the right observed/missing using variance.isinf()
		# in the logistic fragment, this does not occur since there is some tolerance
		# TODO: investigate why this is the case
		# this is a workaround to fetch the observation directly, but it is not the right way to do it
		x = self.children[c].children.values().__iter__().__next__().values.values # N x p
		var = self.parameters["log_variance"].data.exp() # p
		var = var.unsqueeze(0).expand(x.shape[0], -1)
		# when unobserved, we need to send an empty message
		var = torch.where(x.isnan(), torch.full_like(var, torch.inf), var)
		m = torch.where(x.isnan(), torch.zeros_like(var), x)
		self.messages_to_parents[p].message_to_variable = Normal.from_mean_and_variance(m, var)

	def update_parameters(self):
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		# mfp = self.messages_to_parents[p].message_to_factor
		mfc = self.messages_to_children[c].message_to_factor
		mfp = self.parents[p].posterior
		# mfc = self.children[c].posterior
		m, v = mfp.mean_and_variance
		observed = mfc.precision.isinf()
		x = mfc.mean
		s2 = (x - m).pow(2.) + v
		s2sum = torch.where(observed, s2, torch.zeros_like(s2)).sum(dim=0)
		n = observed.sum(dim=0)
		new_log_variance = torch.log(s2sum / n)
		self.parameters["log_variance"].data = new_log_variance

	def forward(self, **kwargs):
		pass

	def elbo(self, x: torch.Tensor | None = None):
		c_id = self._name_to_id["child"]
		p_id = self._name_to_id["parent"]
		if x is None:
			x = self.children[c_id].posterior.mean
			observed = self.children[c_id].posterior.precision.isinf()
			x = torch.where(observed, x, torch.full_like(x, torch.nan))
		m, v = self.parents[p_id].posterior.mean_and_variance
		s2 = self.parameters["log_variance"].exp()
		elbo = (v + m.pow(2.) - 2. * m * x + x.pow(2.)) / s2
		elbo += torch.log(s2).reshape(1, -1) + math.log(2. * math.pi)
		elbo = torch.where(x.isnan(), torch.zeros_like(elbo), elbo)
		return - 0.5 * elbo.sum()

	def log_likelihood(self, x: torch.Tensor | None = None):
		c_id = self._name_to_id["child"]
		p_id = self._name_to_id["parent"]
		if x is None:
			x = self.children[c_id].posterior.mean
			observed = self.children[c_id].posterior.precision.isinf()
			x = torch.where(observed, x, torch.full_like(x, torch.nan))
		m = self.parents[p_id].posterior.mean
		s2 = self.parameters["log_variance"].exp()
		llk = (m - x).pow(2.) / s2
		llk += torch.log(s2).reshape(1, -1) + math.log(2. * math.pi)
		llk = torch.where(x.isnan(), torch.zeros_like(llk), llk)
		return - 0.5 * llk.sum()


class GaussianFactorToParentMessage(Message):

	_name = "GaussianFactorToParentMessage"

	def __init__(self, variable: Variable, factor: GaussianFactor):
		super(GaussianFactorToParentMessage, self).__init__(variable, factor)
		self._message_to_factor = Normal.unit_from_dimension(variable.shape)
		self._message_to_variable = Normal.unit_from_dimension(variable.shape)


class GaussianFactorToChildMessage(Message):

	_name = "GaussianFactorToChildMessage"

	def __init__(self, variable: Variable, factor: GaussianFactor):
		super(GaussianFactorToChildMessage, self).__init__(variable, factor)
		self._message_to_factor = Normal.unit_from_dimension(variable.shape)
		self._message_to_variable = Normal.unit_from_dimension(variable.shape)