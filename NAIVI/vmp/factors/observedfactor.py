from __future__ import annotations
import torch
from .factor import Factor
from ..distributions import Normal, MultivariateNormal, PointMass, Distribution
from ..messages import Message
from ..variables import ObservedVariable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class ObservedFactor(Factor):
	"""Contains observed values, and possibly some missing values."""

	_name = "ObservedFactor"

	def __init__(self, values: torch.Tensor, parent: Variable = None, dist: Distribution = Normal):
		self._dist = dist
		self.values = ObservedVariable(values, dist=dist)
		super(ObservedFactor, self).__init__(parent=parent)
		self.values.set_parents(parent=self)

	def initialize_messages_to_parents(self):
		i = self._name_to_id["parent"]
		self.messages_to_parents[i] = ObservedToParentMessage(self.parents[i], self, self._dist)


class ObservedToParentMessage(Message):

	_name = "ObservedToParentMessage"

	def __init__(self, variable: ObservedVariable, factor: ObservedFactor, dist: Distribution = Normal):
		super(ObservedToParentMessage, self).__init__(variable, factor)
		self._message_to_variable = dist.point_mass(factor.values.values)

	def _set_message_to_variable(self, msg):
		raise RuntimeError("should not be calling this")