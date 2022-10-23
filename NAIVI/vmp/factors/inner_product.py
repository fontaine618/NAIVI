from __future__ import annotations
import torch
from .factor import Factor
from ..distributions.normal import Normal, MultivariateNormal
from ..messages import Message

from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class InnerProduct(Factor):
	"""result = left'right

	The variational approximation is that result is Normal with mean and variance
	computed from moment matching the moments of the product of the two parents.

	In particular,
	mean = ml ' mr
	variance = ml' Vr ml + mr' Vl mr + Tr(Vl Vr)
	"""

	_name = "InnerProduct"
	_deterministic = True

	def __init__(self, left: Variable, right: Variable):
		super(InnerProduct, self).__init__(left=left, right=right)

	def initialize_messages_to_parents(self):
		for i, p in self.parents.items():
			self.messages_to_parents[i] = InnerProductToParentMessage(p, self)

	def initialize_messages_to_children(self):
		i = self._name_to_id["child"]
		self.messages_to_children[i] = InnerProductToChildMessage(self.children[i], self)

	def update_messages_to_children(self):
		l_id = self._name_to_id["left"]
		r_id = self._name_to_id["right"]
		c_id = self._name_to_id["child"]
		mfl = self.messages_to_parents[l_id].message_to_factor
		mfr = self.messages_to_parents[r_id].message_to_factor
		ml, Vl = mfl.mean_and_variance
		mr, Vr = mfr.mean_and_variance
		m = (ml * mr).sum(-1).unsqueeze(-1)
		v = \
			torch.einsum("nk, nkl, nl -> n", ml, Vr, ml) + \
			torch.einsum("nk, nkl, nl -> n", mr, Vl, mr) + \
			torch.einsum("nkl, nkl -> n", Vl, Vr)
		v.unsqueeze_(-1)
		self.messages_to_children[c_id].message_to_variable = Normal.from_mean_and_variance(m, v)

	def update_messages_to_parents(self):
		c_id = self._name_to_id["child"]
		r_id = self._name_to_id["right"]
		l_id = self._name_to_id["left"]
		mfc = self.messages_to_children[c_id].message_to_factor
		mfl = self.messages_to_parents[l_id].message_to_factor
		mfr = self.messages_to_parents[r_id].message_to_factor
		pfc, mtpfc = mfc.precision_and_mean_times_precision
		mfl, Vfl = mfl.mean_and_variance
		mfr, Vfr = mfr.mean_and_variance
		mtptl = mfr * mtpfc
		ptl = pfc.unsqueeze(-1) * (Vfr + mfr.unsqueeze(-1) * mfr.unsqueeze(-2))
		mtptr = mfl * mtpfc
		ptr = pfc.unsqueeze(-1) * (Vfl + mfl.unsqueeze(-1) * mfl.unsqueeze(-2))
		self.messages_to_parents[l_id].message_to_variable = MultivariateNormal(ptl, mtptl)
		self.messages_to_parents[r_id].message_to_variable = MultivariateNormal(ptr, mtptr)


class InnerProductToParentMessage(Message):

	_name = "InnerProductToParentMessage"

	def __init__(self, variable: Variable, factor: InnerProduct):
		super(InnerProductToParentMessage, self).__init__(variable, factor)
		self._message_to_variable = MultivariateNormal.unit_from_dimension(variable.shape)
		self._message_to_factor = MultivariateNormal.unit_from_dimension(variable.shape)


class InnerProductToChildMessage(Message):

	_name = "InnerProductToChildMessage"

	def __init__(self, variable: Variable, factor: InnerProduct):
		super(InnerProductToChildMessage, self).__init__(variable, factor)
		self._message_to_variable = Normal.unit_from_dimension(variable.shape)
		self._message_to_factor = Normal.unit_from_dimension(variable.shape)