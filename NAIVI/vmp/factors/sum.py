from __future__ import annotations
import torch
import torch.distributions
from .factor import Factor
from ..distributions.normal import Normal
from ..distributions import Distribution
from ..messages import Message

from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class Sum(Factor):
	"""Sum multiple parents into a child node.

	Assumes matching dimensions for parents and child. All parents and
	child are assumed to be Normal distributions.
	"""

	_name = "Sum"
	_deterministic = True

	def __init__(self, **kwargs: Variable):
		super(Sum, self).__init__(**kwargs)

	def initialize_messages_to_parents(self):
		for i, parent in self.parents.items():
			self.messages_to_parents[i] = SumToParentMessage(parent, self)

	def initialize_messages_to_children(self):
		i = self._name_to_id["child"]
		self.messages_to_children[i] = SumToChildMessage(self.children[i], self)

	def update_messages_to_children(self):
		i = self._name_to_id["child"]
		means, variances = [], []
		for msg in self.messages_to_parents.values():
			m, v = msg.message_to_factor.mean_and_variance
			means.append(m)
			variances.append(v)
		m, v = sum(means), sum(variances)
		self.messages_to_children[i].message_to_variable = Normal.from_mean_and_variance(m, v)

	def update_messages_to_parents(self):
		i = self._name_to_id["child"]
		mfc, vfc = self.messages_to_children[i].message_to_factor.mean_and_variance
		mfp, vfp = {}, {}
		for i, msg in self.messages_to_parents.items():
			mfp[i], vfp[i] = msg.message_to_factor.mean_and_variance
		smfp, svfp = sum(mfp.values()), sum(vfp.values())
		for i, msg in self.messages_to_parents.items():
			m = mfc - smfp + mfp[i]
			v = vfc + svfp - vfp[i]
			v = torch.where(v.isnan(), float("inf"), v)
			msg.message_to_variable = Normal.from_mean_and_variance(m, v)

	def forward(self, **kwargs):
		c_id = self._name_to_id["child"]
		sc = 0.
		for i, parent in self.parents.items():
			sc += parent.samples
		self.children[c_id].samples = sc


class SumToParentMessage(Message):

	_name = "SumToParentMessage"

	def __init__(self, variable: Variable, factor: Sum):
		super(SumToParentMessage, self).__init__(variable, factor)
		self._message_to_variable = Normal(
			torch.zeros(variable.shape),
			torch.ones(variable.shape)
		)


class SumToChildMessage(Message):

	_name = "SumToChildMessage"

	def __init__(self, variable: Variable, factor: Sum):
		super(SumToChildMessage, self).__init__(variable, factor)
		self._message_to_variable = Normal(
			torch.zeros(variable.shape),
			torch.ones(variable.shape)
		)