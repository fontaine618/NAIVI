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


def _tilted_fixed_point(mean, variance, max_iter=20):
	a = torch.full_like(mean, 0.5)
	for _ in range(max_iter):
		a = torch.sigmoid(mean - (1 - 2 * a) * variance / 2)
	return a


def _ms_expit_moment(degree: int, mean: torch.Tensor, variance: torch.Tensor):
	n = len(mean.shape)
	mean = mean.unsqueeze(-1)
	sd = variance.sqrt()
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
		return (f1 * f2).sum(-1)
	elif degree == 1:
		f1 = p * s / sqrt
		f2 = _std_normal.log_prob(arg).exp()
		return (f1 * f2).sum(-1) * sd
	else:
		raise NotImplementedError("only degree 0 or 1 implemented")


class Logistic(Factor):
	"""
	P[X=1] = sigmoid(mean)
	"""

	_name = "Logistic"

	def __init__(self, dim: int, parent: Variable, method: str = "quadratic"):
		super(Logistic, self).__init__(parent=parent)
		# self.parameters["variance"] = torch.nn.Parameter(torch.ones(dim))
		if method == "quadratic":
			self._update = self._quadratic_update
			self._elbo = self._quadratic_elbo
		elif method == "tilted":
			self._update = self._tilted_update
			i = self._name_to_id["parent"]
			self.messages_to_parents[i].damping = 0.2
			self._elbo = self._tilted_elbo
		elif method == "mk":  # default
			self._update = self._mk_update
			i = self._name_to_id["parent"]
			self.messages_to_parents[i].damping = 0.2
			self._elbo = self._mk_elbo
		else:
			raise ValueError(f"cannot recognize method {method}")

	def initialize_messages_to_parents(self):
		i = self._name_to_id["parent"]
		self.messages_to_parents[i] = LogisticToLogitMessage(self.parents[i], self)

	def initialize_messages_to_children(self):
		i = self._name_to_id["child"]
		self.messages_to_children[i] = LogisticToObservationMessage(self.children[i], self)

	def update_messages_to_children(self):
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		mfp = self.messages_to_parents[p_id].message_to_factor
		m, v = mfp.mean_and_variance
		proba = _ms_expit_moment(0, m, v)
		self.messages_to_children[c_id].message_to_variable = Probability(proba)

	def update_messages_to_parents(self):
		self._update()

	def elbo(self):
		return self._elbo()

	def _quadratic_elbo(self):
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		mfc = self.children[c_id].posterior
		mfp = self.parents[p_id].posterior
		m, v = mfp.mean_and_variance
		s = mfc.proba - 0.5
		t = (m.pow(2.) + v).sqrt()
		elbo = torch.sigmoid(t).log() + s * m - t * 0.5
		elbo = torch.where(s.abs()==0.5, elbo, torch.zeros_like(elbo))
		return elbo.sum()

	def _mk_elbo(self):
		# TODO: implement this
		pass

	def _tilted_elbo(self):
		# TODO: implement this
		pass

	def _mk_update(self):
		# TODO seems incorrect: maybe use the posterior as the message?
		"""Uses Knowles & Minka (2011)"""
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		mfc = self.messages_to_children[c_id].message_to_factor
		# mfp = self.messages_to_parents[p_id].message_to_factor
		mfp = self.parents[p_id].posterior
		m, v = mfp.mean_and_variance
		p, mtp = mfp.precision_and_mean_times_precision
		x = mfc.proba
		b0 = _ms_expit_moment(0, m, v)
		b1 = _ms_expit_moment(1, m, v)
		# Knowles and Minka (2011, Section 5.2)
		m1 = b1 * v.sqrt() + m * b0
		p1 = (m1 - m * b0) * p
		mtp1 = m * p1 + x - b0
		# Nolan and Wand (2017, Eq. 6) I think it's equivalent to the above
		# p1 = b1 * p.sqrt()
		# mtp1 = m * p1 + x - b0
		self.messages_to_parents[p_id].message_to_variable = Normal(p1, mtp1)

	def _quadratic_update(self):
		"""Uses the quadratic bound of Jaakola and Jordan (2000)"""
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		mfc = self.messages_to_children[c_id].message_to_factor
		mfp = self.messages_to_parents[p_id].message_to_factor
		m, v = mfp.mean_and_variance
		s = mfc.proba - 0.5
		t = (m.pow(2.) + v).sqrt()
		lam = (torch.sigmoid(t) - 0.5) / t
		p1 = lam
		mtp1 = s
		p1 = torch.where(s == 0., torch.zeros_like(p1), p1)  # mtp1 should be fine already
		self.messages_to_parents[p_id].message_to_variable = Normal(p1, mtp1)

	def _tilted_update(self):
		"""Uses the tilted bound of Saul and Jordan (1999)"""
		# TODO seems incorrect: maybe use the posterior as the message?
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		mfc = self.messages_to_children[c_id].message_to_factor
		# mfp = self.messages_to_parents[p_id].message_to_factor
		mfp = self.parents[p_id].posterior
		m, v = mfp.mean_and_variance
		x = mfc.proba
		a = _tilted_fixed_point(m, v)
		p1 = a * (1 - a)
		mtp1 = m * p1 + x - a
		self.messages_to_parents[p_id].message_to_variable = Normal(p1, mtp1)

	def elbo_mc(self, n_samples: int = 1):
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		sp = self.parents[p_id].sample(n_samples)  # B x ...
		sc = self.children[c_id].sample(n_samples)  # B x ...
		# sc has no missing values, need to fetch them back
		proba = self.messages_to_children[c_id].message_to_factor.proba
		# proba should only contain 0., 0.5 or 1.
		obs = torch.logical_or(proba.lt(1.e-10), proba.gt(1.-1.e-10))
		sc = torch.where(obs, sc, torch.full_like(sc, float("nan")))
		elbo = torch.sigmoid((2. * sc -1.) * sp).log()
		return elbo.nansum(dim=(-1, -2)).nanmean(0)


class LogisticToLogitMessage(Message):

	_name = "LogisticToLogitMessage"

	def __init__(self, variable: Variable, factor: Logistic, damping: float = 1.):
		super(LogisticToLogitMessage, self).__init__(variable, factor, damping=damping)
		self._message_to_factor = Normal.unit_from_dimension(variable.shape)
		self._message_to_variable = Normal.unit_from_dimension(variable.shape)


class LogisticToObservationMessage(Message):

	_name = "LogisticToObservationMessage"

	def __init__(self, variable: Variable, factor: Logistic):
		super(LogisticToObservationMessage, self).__init__(variable, factor)
		self._message_to_factor = Probability.unit_from_dimension(variable.shape)
		self._message_to_variable = Probability.unit_from_dimension(variable.shape)