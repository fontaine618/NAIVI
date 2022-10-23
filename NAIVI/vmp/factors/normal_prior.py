from __future__ import annotations
import torch
from .factor import Factor
from ..distributions.normal import Normal, MultivariateNormal, _batch_mahalanobis
from ..messages import Message

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class NormalPrior(Factor):
	"""
	y ~ N(mu, sigma^2).

	No parents, could have multiple children, but in principle this should only be one.
	"""

	_name = "NormalPrior"

	def __init__(self, mean=0., variance=1.):
		super(NormalPrior, self).__init__()
		self.parameters["mean"] = torch.nn.Parameter(torch.Tensor([mean]), requires_grad=False)
		self.parameters["variance"] = torch.nn.Parameter(torch.Tensor([variance]), requires_grad=False)

	def initialize_messages_to_children(self):
		for n, c in self.children.items():
			self.messages_to_children[n] = NormalPriorToChildMessage(variable=c, factor=self)

	def elbo(self):
		m0, v0 = self.parameters["mean"].data, self.parameters["variance"].data
		i = self._name_to_id["child"]
		mq, vq = self.children[i].posterior.mean_and_variance
		elbo = (vq/v0).log() + 1. - (mq.pow(2.) + vq - 2. * m0*mq + m0**2) / v0
		return elbo.sum() * 0.5


class NormalPriorToChildMessage(Message):

	_name = "NormalPriorToChildMessage"

	def __init__(self, variable: Variable, factor: NormalPrior):
		super(NormalPriorToChildMessage, self).__init__(variable, factor)
		dim = variable.shape
		mean = torch.full(dim, factor.get("mean").item())
		variance = torch.full(dim, factor.get("variance").item())
		self.message_to_variable = Normal.from_mean_and_variance(mean, variance)


class MultivariateNormalPrior(Factor):
	"""
	y ~ N(mean, variance * I).

	No parents, could have multiple children, but in principle this should only be one.
	No need to do updates, though we could implement message to parent eventually.
	"""

	_name = "MultivariateNormalPrior"

	def __init__(self, mean=0., variance=1., dim=1):
		super(MultivariateNormalPrior, self).__init__()
		self.parameters["mean"] = torch.nn.Parameter(torch.Tensor([mean]), requires_grad=False)
		self.parameters["variance"] = torch.nn.Parameter(torch.Tensor([variance]), requires_grad=False)
		self.dim = dim

	def initialize_messages_to_children(self):
		for n, c in self.children.items():
			self.messages_to_children[n] = MultivariateNormalPriorToChildMessage(variable=c, factor=self)

	def elbo(self):
		m0, v0 = self.parameters["mean"].data, self.parameters["variance"].data
		m0 = m0.expand(self.dim)
		det0 = v0**self.dim
		p0 = torch.eye(self.dim) / v0
		i = self._name_to_id["child"]
		mq, vq = self.children[i].posterior.mean_and_variance
		detq = vq.det().sum()
		trace = (p0 * vq).sum()
		diff = m0 - mq
		quad = torch.einsum("ik, kj, ij->i", diff, p0, diff).sum()
		elbo = - trace - quad + self.dim + detq.log() - det0.log()
		return elbo.sum() * 0.5


class MultivariateNormalPriorToChildMessage(Message):

	_name = "MultivariateNormalPriorToChildMessage"

	def __init__(self, variable: Variable, factor: MultivariateNormalPrior):
		super(MultivariateNormalPriorToChildMessage, self).__init__(variable, factor)
		dim = variable.shape
		mean = torch.full(dim, factor.get("mean").item())
		variance = torch.eye(factor.dim).expand(*dim, factor.dim) * factor.get("variance")
		self._message_to_variable = MultivariateNormal.from_mean_and_variance(mean, variance)
		self._message_to_factor = MultivariateNormal.unit_from_dimension(dim=variable.shape)