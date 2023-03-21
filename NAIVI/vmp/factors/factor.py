from __future__ import annotations
import torch.nn
import itertools

from .. import VMP_OPTIONS
from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class Factor:
	"""A factor and all the messages."""

	_name = "Factor"
	_deterministic = False
	new_id = itertools.count()
	instance = dict()

	def __init__(self, **kwargs: Variable):
		self.id: int = next(Factor.new_id)
		self.children: Dict[int, Variable] = {}
		if not hasattr(self, "parameters"):
			# some factors will initialize the parameters beforehand
			self.parameters: Dict[str, torch.nn.Parameter] = {}
		self.parents: Dict[int, Variable] = {}
		self.messages_to_parents: Dict[int, Message] = {}
		self.messages_to_children: Dict[int, Message] = {}
		self._name_to_id: Dict[str, int] = {}
		self._id_to_name: Dict[int, str] = {}
		self.set_parents(**kwargs)
		Factor.instance[self.id] = self
		if VMP_OPTIONS["logging"]: print(f"Initialized {repr(self)}")

	@property
	def name(self):
		return self._name

	@property
	def deterministic(self):
		return self._deterministic

	def initialize_messages_to_parents(self):
		pass

	def initialize_messages_to_children(self):
		pass

	def elbo(self, **kwargs):
		"""If not reimplemented, we assume there is no elbo contribution from this factor."""
		return torch.Tensor([0.])

	def elbo_mc(self, n_samples: int = 1):
		"""Approximates ELBO using MC from the samples.

		If not implemented, simply returns the elbo.
		This should be done when the elbo is exact."""
		return self.elbo()

	def update_messages_from_children(self):
		for i, child in self.children.items():
			mtv = self.messages_to_children[i].message_to_variable
			post = child.posterior
			self.messages_to_children[i].message_to_factor = post / mtv

	def update_messages_from_parents(self):
		for i, parent in self.parents.items():
			mtp = self.messages_to_parents[i].message_to_variable
			post = parent.posterior
			self.messages_to_parents[i].message_to_factor = post / mtp

	def update_messages_to_children(self):
		pass

	def update_messages_to_parents(self):
		pass

	def update_parameters(self):
		pass

	def set_parents(self, **kwargs: Variable):
		for k, v in kwargs.items():
			self.parents[v.id] = v
			self._name_to_id[k] = v.id
			self._id_to_name[v.id] = k
			v.set_children(child=self)
		self.initialize_messages_to_parents()

	def set_children(self, **kwargs: Variable):
		for k, v in kwargs.items():
			self.children[v.id] = v
			self._name_to_id[k] = v.id
			self._id_to_name[v.id] = k
			v.set_parents(parent=self)
		self.initialize_messages_to_children()

	def get(self, parameter: str) -> torch.Tensor:
		return self.parameters[parameter].data

	def forward(self, n_samples: int = 1):
		"""Takes parents.sample and updates children.sample"""
		pass

	def __repr__(self):
		return f"[f{self.id}] {self._name}"

	def __str__(self):
		out = repr(self) + "\n"
		out += "     Parameters:\n"
		for n, p in self.parameters.items():
			dim_str = ", ".join([str(d) for d in p.shape])
			out += f"     - {n}: dim=({dim_str})\n"
		out += "     Parents:\n"
		for i, p in self.parents.items():
			out += f"     - {self._id_to_name[i]}: {repr(p)}\n"
			out += f"       Messages: {repr(self.messages_to_parents[i])}\n"
			out += f"       - To factor:   {repr(self.messages_to_parents[i].message_to_factor)}\n"
			out += f"       - To variable: {repr(self.messages_to_parents[i].message_to_variable)}\n"
		out += "     Children:\n"
		for i, p in self.children.items():
			out += f"     - {self._id_to_name[i]}: {repr(p)}\n"
			out += f"       Messages: {repr(self.messages_to_children[i])}\n"
			out += f"       - To factor:   {repr(self.messages_to_children[i].message_to_factor)}\n"
			out += f"       - To variable: {repr(self.messages_to_children[i].message_to_variable)}\n"
		return out