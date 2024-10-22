from __future__ import annotations
import torch
import math
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
		# kl = (vq/v0).log() + 1. - (mq.pow(2.) + vq - 2. * m0*mq + m0**2) / v0
		kl = (v0/vq).log() - 1. + (mq - m0).pow(2.) / v0 + vq / v0
		return - kl.sum() * 0.5

	# def forward(self, n_samples: int = 1):
	# 	c_id = self._name_to_id["child"]
	# 	dim = tuple(self.children[c_id].shape)
	# 	dim = n_samples, *dim
	# 	m0, v0 = self.parameters["mean"].data.item(), self.parameters["variance"].data.item()
	# 	sc = torch.full(dim, m0) + torch.randn(dim) * torch.sqrt(torch.full(dim, v0))
	# 	self.children[c_id].sample = sc


class NormalPriorToChildMessage(Message):

	_name = "NormalPriorToChildMessage"

	def __init__(self, variable: Variable, factor: NormalPrior):
		super(NormalPriorToChildMessage, self).__init__(variable, factor)
		dim = variable.shape
		mean = torch.full(dim, factor.get("mean").item())
		variance = torch.full(dim, factor.get("variance").item())
		self._message_to_variable = Normal.from_mean_and_variance(mean, variance)


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
		logdet0 = math.log(v0 * 2 * math.pi) * self.dim
		p0 = torch.eye(self.dim) / v0
		i = self._name_to_id["child"]
		mq, vq = self.children[i].posterior.mean_and_variance
		logdetq = (vq * 2 * math.pi).logdet()
		trace = (p0.unsqueeze(0) * vq).sum((-1, -2))
		diff = m0 - mq
		quad = torch.einsum("ik, kj, ij->i", diff, p0, diff)
		kl = trace + quad - logdetq + logdet0 - self.dim
		if kl.isnan().any():
			which = kl.isnan()
		return - 0.5 * kl.sum()

	# def elbo_mc(self):
	# 	return self.elbo()
		# # NB the above is exact, so this is just to check
		# c_id = self._name_to_id["child"]
		# m0, v0 = self.parameters["mean"].data.item(), self.parameters["variance"].data.item()
		# sc = self.children[c_id].samples
		# # cross entropy with prior
		# logdet0 = math.log(v0 * 2 * math.pi) * self.dim
		# ce0 = (sc - m0).pow(2.).sum(-1) / v0
		# ce0 += logdet0
		# # entropy of posterior
		# mq, vq = self.children[c_id].posterior.mean_and_variance
		# pq = self.children[c_id].posterior.precision
		# logdetq = (vq * 2 * math.pi).logdet()
		# diff = sc - mq.unsqueeze(0)
		# quad = torch.einsum("mij, ikj, mik->mi", diff, pq, diff)
		# eq = logdetq + quad
		# # sum both
		# elbo = ce0 - eq
		# return 0.5 * elbo.sum(dim=-1).mean(dim=0)


class MultivariateNormalPriorToChildMessage(Message):

	_name = "MultivariateNormalPriorToChildMessage"

	def __init__(self, variable: Variable, factor: MultivariateNormalPrior):
		super(MultivariateNormalPriorToChildMessage, self).__init__(variable, factor)
		dim = variable.shape
		mean = torch.full(dim, factor.get("mean").item())
		variance = torch.eye(factor.dim).expand(*dim, factor.dim) * factor.get("variance")
		self._message_to_variable = MultivariateNormal.from_mean_and_variance(mean, variance)
		self._message_to_factor = MultivariateNormal.unit_from_dimension(dim=variable.shape)