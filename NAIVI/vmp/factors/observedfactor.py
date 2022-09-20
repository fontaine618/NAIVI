from __future__ import annotations
import torch
from .factor import Factor
from ..distributions.normal import Normal, MultivariateNormal
from ..distributions.point_mass import PointMass
from ..messages import Message
from ..variables import ObservedVariable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class ObservedFactor(Factor):
	"""Contains observed values, and possibly some missing values."""

	_name = "ObservedFactor"

	def __init__(self, values: torch.Tensor, parent: Variable = None):
		self.values = ObservedVariable(values)
		super(ObservedFactor, self).__init__(parent=parent)
		self.values.set_parents(parent=self)

	def initialize_messages_to_parents(self):
		i = self._name_to_id["parent"]
		self.messages_to_parents[i] = ObservedToParentMessage(self.values, self)


class ObservedToParentMessage(Message):

	_name = "ObservedToParentMessage"

	def __init__(self, variable: ObservedVariable, factor: ObservedFactor):
		super(ObservedToParentMessage, self).__init__(variable, factor)
		self._message_to_variable = PointMass(variable.values)

	def _set_message_to_variable(self, msg):
		raise RuntimeError("should not be calling this")