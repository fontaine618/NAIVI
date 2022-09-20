from __future__ import annotations
import torch
import torch.distributions
from .factor import Factor
from ..distributions.normal import Normal
from ..distributions.probability import Probability
from ..messages import Message

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message

_p = torch.tensor([
	0.003246343272134,
	0.051517477033972,
	0.195077912673858,
	0.315569823632818,
	0.274149576158423,
	0.131076880695470,
	0.027912418727972,
	0.001449567805354
])

_s = torch.tensor([
	1.365340806296348,
	1.059523971016916,
	0.830791313765644,
	0.650732166639391,
	0.508135425366489,
	0.396313345166341,
	0.308904252267995,
	0.238212616409306
])

_std_normal = torch.distributions.normal.Normal(0., 1.)


def _ms_expit_moment(degree: int, mean: torch.Tensor, variance: torch.Tensor):
	n = len(mean.shape)
	mean = mean.unsqueeze(-1)
	variance = variance.unsqueeze(-1)
	p = _p
	s = _s
	for _ in range(n):
		p = p.unsqueeze(0)
		s = s.unsqueeze(0)
	sqrt = (1 + variance * s.pow(2)).sqrt()
	arg = mean * s / sqrt
	if degree == 0:
		f1 = p
		f2 = _std_normal.cdf(arg)
	elif degree == 1:
		f1 = p * s / sqrt
		f2 = _std_normal.log_prob(arg).exp()
	else:
		raise NotImplementedError("only degree 0 or 1 implemented")
	return (f1 * f2).sum(-1)


class Logistic(Factor):
	"""
	P[X=1] = sigmoid(mean)
	"""

	_name = "Logistic"

	def __init__(self, dim: int, parent: Variable, method: str = "MK"):
		super(Logistic, self).__init__(parent=parent)
		self.parameters["variance"] = torch.nn.Parameter(torch.ones(dim))
		if method == "quadratic":
			self._update = self._quadratic_update
		elif method == "tilted":
			self._update = self._tilted_update
		else:  # default
			self._update = self._mk_update

	def initialize_messages_to_parents(self):
		i = self._name_to_id["parent"]
		self.messages_to_parents[i] = LogisticToParentMessage(self.parents[i], self)

	def initialize_messages_to_children(self):
		i = self._name_to_id["child"]
		self.messages_to_children[i] = LogisticToChildMessage(self.children[i], self)

	def update_messages_to_children(self):
		pass

	def update_messages_to_parents(self):
		self._update()

	def _mk_update(self):
		"""Uses Knowles & Minka (2011)"""
		pass

	def _quadratic_update(self):
		"""Uses the quadratic bound of Jaakola and Jordan (2000)"""
		pass

	def _tilted_update(self):
		"""Uses the tilted bound of Saul and Jordan (1999)"""
		pass


class LogisticToParentMessage(Message):

	_name = "LogisticToParentMessage"

	def __init__(self, variable: Variable, factor: Logistic):
		super(LogisticToParentMessage, self).__init__(variable, factor)
		self._message_to_factor = Normal.unit_from_dimension(variable.shape)
		self._message_to_variable = Normal.unit_from_dimension(variable.shape)


class LogisticToChildMessage(Message):

	_name = "LogisticToChildMessage"

	def __init__(self, variable: Variable, factor: Logistic):
		super(LogisticToChildMessage, self).__init__(variable, factor)
		self._message_to_factor = Probability.unit_from_dimension(variable.shape)
		self._message_to_variable = Probability.unit_from_dimension(variable.shape)