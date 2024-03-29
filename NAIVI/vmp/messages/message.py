from __future__ import annotations
import itertools
import torch

from .. import VMP_OPTIONS
from ..distributions import Unit, Normal

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

	def __init__(self, variable: Variable, factor: Factor, damping: float = 1., **kw):
		self._factor_is_deterministic = factor.deterministic
		self.id = next(Message.new_id)
		self.variable = variable
		self.factor = factor
		self._dim = variable.shape
		self._message_to_factor: Distribution = Normal.unit_from_dimension(self._dim)
		self._message_to_variable: Distribution = Normal.unit_from_dimension(self._dim)
		self.damping = damping
		Message.instance[self.id] = self
		if VMP_OPTIONS["logging"]: print(f"Initialized {repr(self)}")

	@property
	def name(self):
		return self._name

	def __repr__(self):
		return f"[m{self.id}] {self._name}"

	def _set_message_to_variable(self, msg: Distribution):
		"""Stores the new message and updates the posterior of the variable."""
		if VMP_OPTIONS["logging"]: print(f"Update message from {repr(self.factor)} to {self.variable} ({repr(self)})")
		prev_msg = self._message_to_variable
		# self._message_to_variable = msg
		# # for damping, we store the full message, but the posterior update is only partial
		# self.variable.update(prev_msg ** self.damping, self._message_to_variable ** self.damping)

		if self.damping != 1.:
			# This might not work for aggreagted messages, but
			# I'm leaving it like this for now, since I only plan to use damping
			# for the logistic fragments
			self._message_to_variable = (msg**self.damping) * (prev_msg**(1-self.damping))
		else:
			self._message_to_variable = msg
		self.variable.update(prev_msg, self._message_to_variable)

	def _get_message_to_variable(self):
		return self._message_to_variable

	def _del_message_to_variable(self):
		del self._message_to_variable

	message_to_variable = property(
		_get_message_to_variable,
		_set_message_to_variable,
		_del_message_to_variable
	)

	def _set_message_to_factor(self, msg: Distribution):
		if VMP_OPTIONS["logging"]: print(f"Update message from {self.variable} to {repr(self.factor)}")
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

