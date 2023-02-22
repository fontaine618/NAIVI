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
		# this is not the standard VMP message,
		# we rather send this message so that the posterior is the predictive posterior
		# for missing values
		# This is also fine to do, because, the message_to_factor below will
		# be fine since it removes the message in the other direction and only return the
		# observed/missing message
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfp = self.messages_to_parents[p].message_to_factor # N x p
		# mfp = self.parents[p].posterior # N x p
		m, v = mfp.mean_and_variance # N x p, N x p
		var = self.parameters["log_variance"].data.exp() # p
		v = v + var.reshape((1, -1))
		self.messages_to_children[c].message_to_variable = Normal.from_mean_and_variance(m, v)

	def update_messages_to_parents(self):
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		# now, the child is derived (equal to its obervation), so we just take the message
		# however, we can have unobserved values so we need to adjust the variance below
		mfc = self.messages_to_children[c].message_to_factor # N x p
		m, v = mfc.mean_and_variance # N x p, N x p
		var = self.parameters["log_variance"].data.exp() # p
		# v = v + var.reshape((1, -1))
		var = var.unsqueeze(0).expand(m.shape(0), -1)
		var = torch.where(v.isinf(), torch.full_like(var, torch.inf), var)
		m = torch.where(v.isinf(), torch.zeros_like(var), m)
		self.messages_to_parents[p].message_to_variable = Normal.from_mean_and_variance(m, v)

	def update_parameters(self):
		# TODO: check this
		# curious to see the ELBO comparison with a gradient update
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		# mfp = self.messages_to_parents[p].message_to_factor
		# mfc = self.messages_to_children[c].message_to_factor
		mfp = self.parents[p].posterior
		mfc = self.children[c].posterior
		m, v = mfp.mean_and_variance
		observed = mfc.precision > 0.
		x = mfc.mean
		s2 = (x - m).pow(2.) + v
		s2sum = torch.where(observed, s2, torch.zeros_like(s2)).sum(dim=0)
		n = observed.sum(dim=0)
		self.parameters["log_variance"].data = torch.log(s2sum / n)

	def elbo(self):
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		m, v = self.parents[p].posterior.mean_and_variance
		x = self.children[c].posterior.mean
		observed = self.children[c].posterior.precision > 0.
		s2 = self.parameters["log_variance"].exp()
		elbo = (v + m.pow(2.) - 2. * m * x + x.pow(2.)) / s2
		elbo += torch.log(s2).reshape(1, -1) + math.log(2. * math.pi)
		elbo = torch.where(observed, elbo, torch.zeros_like(elbo)).sum(dim=0)
		return - 0.5 * elbo.sum()

	def forward(self, **kwargs):
		pass
		# This should be the update, but here the child is observed, so we don't update it
		# p_id = self._name_to_id["parent"]
		# c_id = self._name_to_id["child"]
		# sp = self.parents[p_id].sample
		# s2 = self.parameters["log_variance"].exp()
		# sc = sp + torch.randn_like(sp) * s2.sqrt()
		# self.children[c_id].sample = sc

	# exact elbo, so we do not need this
	# def elbo_mc(self):
	# 	p_id = self._name_to_id["parent"]
	# 	c_id = self._name_to_id["child"]
	# 	sp = self.parents[p_id].samples
	# 	s2 = self.parameters["log_variance"].exp()
	# 	sc = self.children[c_id].samples
	# 	d = (sc - sp).pow(2.) / s2
	# 	d += torch.log(s2).reshape(1, 1, -1) + math.log(2. * math.pi)
	# 	return - 0.5 * d.nansum(dim=(-1, -2)).nanmean(dim=0)



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