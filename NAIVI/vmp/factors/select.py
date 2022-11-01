from __future__ import annotations
import torch
import torch.distributions
from .factor import Factor
from ..distributions.normal import Normal
from ..distributions import Distribution
from ..messages import Message
from .. import VMP_OPTIONS

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class Select(Factor):
	"""Selects a subset of the input variables.

	Assumes that index corresponds to the first dimension of parent.

	Messages are just copies of the neighbors, the only thing we need to do is to
	perform the aggregation when sending to the parent.
	"""

	_name = "Select"

	def __init__(self, indices: torch.Tensor, parent: Variable = None, dist: type[Normal] = Normal):
		self.indices = indices
		self._dist: type[Normal] = dist
		super(Select, self).__init__(parent=parent)

	def initialize_messages_to_parents(self):
		i = self._name_to_id["parent"]
		self.messages_to_parents[i] = SelectToParentMessage(self.parents[i], self, self.indices, self._dist)

	def initialize_messages_to_children(self):
		i = self._name_to_id["child"]
		self.messages_to_children[i] = SelectToChildMessage(self.children[i], self, self.indices, self._dist)

	def update_messages_from_parents(self):
		for i, parent in self.parents.items():
			mtp = self.messages_to_parents[i]._message_to_variable  # we need all messages here
			post = parent.posterior.index_select(0, self.indices)
			self.messages_to_parents[i].message_to_factor = post / mtp

	def update_messages_to_children(self):
		# just copy the message from the variable
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfp = self.messages_to_parents[p].message_to_factor
		self.messages_to_children[c].message_to_variable = mfp.clone()

	def update_messages_to_parents(self):
		# just copy the message from the variable
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfc = self.messages_to_children[c].message_to_factor
		self.messages_to_parents[p].message_to_variable = mfc.clone()

	def forward(self, **kwargs):
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		sp = self.parents[p_id].samples  # B x E X K, e in [N]
		self.children[c_id].samples = sp.index_select(1, self.indices)


class SelectToParentMessage(Message):
	"""Dispatch message."""

	_name = "SelectToParentMessage"

	def __init__(self, variable: Variable, factor: Select, indices: torch.Tensor, dist: type[Normal]):
		super(SelectToParentMessage, self).__init__(factor=factor, variable=variable)
		self.indices = indices
		dim = torch.Size([len(indices), *variable.shape[1:]])
		self._dim = dim
		self._max_index = variable.shape[0]
		self._message_to_variable = dist.unit_from_dimension(dim)
		self._message_to_factor = dist.unit_from_dimension(dim)

	def _set_message_to_variable(self, msg: Normal):
		if VMP_OPTIONS["logging"]: print(f"Update message from {repr(self.factor)} to {self.variable}")
		prev_msg = self.message_to_variable
		self._message_to_variable = msg
		agg_msg = msg.index_sum(0, self.indices, self._max_index)
		self.variable.update(prev_msg, agg_msg)

	def _get_message_to_variable(self):
		return self._message_to_variable.index_sum(0, self.indices, self._max_index)

	def _del_message_to_variable(self):
		del self._message_to_factor

	message_to_variable = property(
		_get_message_to_variable,
		_set_message_to_variable,
		_del_message_to_variable
	)


class SelectToChildMessage(Message):

	_name = "SelectToChildMessage"

	def __init__(self, variable: Variable, factor: Select, indices: torch.Tensor, dist: type[Normal]):
		super(SelectToChildMessage, self).__init__(factor=factor, variable=variable)
		self.indices = indices
		dim = list(variable.shape)
		dim[0] = len(indices)
		self._dim = dim
		self._message_to_variable = dist.unit_from_dimension(dim)
		self._message_to_factor = dist.unit_from_dimension(dim)