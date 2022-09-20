from __future__ import annotations
import itertools

from ..distributions.point_mass import Unit

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..variables import Variable
	from ..factors import Factor
	from ..distributions import Distribution


class Message:
	"""Container for the two messages between a variable and a factor."""

	_name = "Message"
	new_id = itertools.count()
	instance = dict()

	def __init__(self, variable: Variable, factor: Factor, **kw):
		self.id = next(Message.new_id)
		self.variable = variable
		self.factor = factor
		self._dim = variable.shape
		self._message_to_factor: Distribution = Unit(self._dim)
		self._message_to_variable: Distribution = Unit(self._dim)
		Message.instance[self.id] = self

	def __repr__(self):
		return f"[{self.id:>2}] {self._name}"

	def _set_message_to_variable(self, msg):
		"""Stores the new message and updates the posterior of the variable."""
		prev_msg = self._message_to_variable
		self._message_to_variable = msg
		self.variable.update(prev_msg, msg)

	def _get_message_to_variable(self):
		return self._message_to_variable

	def _del_message_to_variable(self):
		del self._message_to_variable

	message_to_variable = property(
		_get_message_to_variable,
		_set_message_to_variable,
		_del_message_to_variable
	)

	def _set_message_to_factor(self, msg):
		self._message_to_factor = msg

	def _get_message_to_factor(self):
		return self._message_to_factor

	def _del_message_to_factor(self):
		del self._message_to_factor

	message_to_factor = property(
		_get_message_to_factor,
		_set_message_to_factor,
		_del_message_to_factor
	)

