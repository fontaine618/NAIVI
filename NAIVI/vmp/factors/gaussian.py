from __future__ import annotations
import torch
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
		self.parameters["variance"] = torch.nn.Parameter(torch.ones(dim))

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
		m, v = mfp.mean_and_variance # N x p, N x p
		var = self.parameters["variance"].data # p
		v = v + var.reshape((1, -1))
		self.messages_to_children[c].message_to_variable = Normal.from_mean_and_variance(m, v)

	def update_messages_to_parents(self):
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfp = self.messages_to_children[c].message_to_factor # N x p
		m, v = mfp.mean_and_variance # N x p, N x p
		var = self.parameters["variance"].data # p
		v = v + var.reshape((1, -1))
		self.messages_to_parents[p].message_to_variable = Normal.from_mean_and_variance(m, v)


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