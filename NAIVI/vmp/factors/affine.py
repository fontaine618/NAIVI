from __future__ import annotations
import torch
from .factor import Factor
from ..distributions.normal import Normal, MultivariateNormal
from ..messages import Message

from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class Affine(Factor):
	"""child = bias + weight'parent"""

	_name = "Affine"
	_deterministic = True

	def __init__(self, dim_in: int, dim_out: int, parent: Variable):
		self.parameters: Dict[str, torch.nn.Parameter] = dict()
		self.parameters["bias"] = torch.nn.Parameter(torch.randn(dim_out))
		self.parameters["weights"] = torch.nn.Parameter(torch.randn((dim_in, dim_out)))
		super(Affine, self).__init__(parent=parent)

	def initialize_messages_to_parents(self):
		i = self._name_to_id["parent"]
		self.messages_to_parents[i] = AffineToParentMessage(self.parents[i], self)

	def initialize_messages_to_children(self):
		i = self._name_to_id["child"]
		self.messages_to_children[i] = AffineToChildMessage(self.children[i], self)

	def update_messages_to_children(self):
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfp = self.messages_to_parents[p].message_to_factor  # N x p x K
		b = self.parameters["bias"].data  # p
		w = self.parameters["weights"].data  # K x p
		m, v = mfp.mean_and_variance  # N x p x k, N x p x K x K
		mean = torch.einsum("j, kj, ijk -> ij", b, w, m)  # N x p
		variance = torch.einsum("kj, ijkl, lj -> ij", w, v, w)  # N x p
		self.messages_to_children[c].message_to_variable = Normal.from_mean_and_variance(mean, variance)

	def update_messages_to_parents(self):
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfc = self.messages_to_children[c].message_to_factor  # N x p
		m, v = mfc.mean_and_variance  # N x p, N x p
		m = torch.where(m.isnan(), 0., m)
		b = self.parameters["bias"].data  # p
		w = self.parameters["weights"].data  # K x p
		mmb = m - b.reshape((1, -1))  # N x p
		prec = torch.einsum("kj, lj, ij -> ijkl", w, w, 1/v)  # N x p x K x K
		mtp = torch.einsum("ij, kj, ij -> ijk", mmb, w, 1/v)  # N x p x K
		self.messages_to_parents[p].message_to_variable = MultivariateNormal(prec, mtp)

	def update_messages_from_parents(self):
		for i, parent in self.parents.items():
			mtp = self.messages_to_parents[i]._message_to_variable  # we need all messages here
			post = parent.posterior.unsqueeze(1)
			self.messages_to_parents[i].message_to_factor = post / mtp


class AffineToParentMessage(Message):

	_name = "AffineToParentMessage"

	def __init__(self, variable: Variable, factor: Affine):
		super(AffineToParentMessage, self).__init__(
			variable=variable,
			factor=factor
		)
		dim = variable.shape
		dim_parallel = factor.parameters["weights"].shape[1]
		self._dim = torch.Size([dim[0], dim_parallel, *dim[1:]])
		self._message_to_factor = MultivariateNormal.unit_from_dimension(self._dim)
		self._message_to_variable = MultivariateNormal.unit_from_dimension(self._dim)

	def _get_message_to_variable(self):
		return self._message_to_variable.prod(1)

	def _set_message_to_variable(self, msg):
		"""Stores the new message and updates the posterior of the variable."""
		# print(f"Update message from {repr(self.factor)} to {self.variable}")
		prev_msg = self.message_to_variable
		self._message_to_variable = msg
		self.variable.update(prev_msg, msg.prod(1))

	def _del_message_to_variable(self):
		del self._message_to_variable

	message_to_variable = property(
		_get_message_to_variable,
		_set_message_to_variable,
		_del_message_to_variable
	)


class AffineToChildMessage(Message):

	_name = "AffineToChildMessage"

	def __init__(self, variable: Variable, factor: Affine):
		super(AffineToChildMessage, self).__init__(variable, factor)
		self._message_to_factor = Normal.unit_from_dimension(variable.shape)
		self._message_to_variable = Normal.unit_from_dimension(variable.shape)