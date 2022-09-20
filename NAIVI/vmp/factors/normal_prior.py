from __future__ import annotations
import torch
from .factor import Factor
from ..distributions.normal import Normal, MultivariateNormal
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


class MultivariateNormalPriorToChildMessage(Message):

	_name = "MultivariateNormalPriorToChildMessage"

	def __init__(self, variable: Variable, factor: MultivariateNormalPrior):
		super(MultivariateNormalPriorToChildMessage, self).__init__(variable, factor)
		dim = variable.shape
		mean = torch.full(dim, factor.get("mean").item())
		variance = torch.eye(factor.dim).expand(*dim, factor.dim) * factor.get("variance")
		self._message_to_variable = MultivariateNormal.from_mean_and_variance(mean, variance)