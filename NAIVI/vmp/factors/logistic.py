from __future__ import annotations

from itertools import product
from functools import reduce
import torch
import torch.distributions
from .factor import Factor
from .logistic_utils import _gh_quadrature, _log1p_exp, _tilted_fixed_point, _ms_expit_moment
from ..distributions.normal import Normal
from ..distributions.probability import Probability
from ..messages import Message

from typing import TYPE_CHECKING
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class Logistic(Factor):
	"""
	P[X=1] = sigmoid(mean)
	"""

	_name = "Logistic"

	def __init__(self, dim: int, parent: Variable, method: str = "quadratic"):
		super(Logistic, self).__init__(parent=parent)
		self._elbo = self._quadrature_elbo
		if reduce(lambda x, y: x*y, parent.shape) > 5e5:
			self._elbo = self._quadratic_elbo
		self._elbo = self._quadratic_elbo
		self._n_updates = -1
		i = self._name_to_id["parent"]
		if method == "quadratic":
			self._update = self._quadratic_update
			self.messages_to_parents[i]._damping = 1.
		elif method == "adaptive":
			self._update = self._adaptive_update
		elif method == "tilted":
			self._update = self._tilted_update
			# self.messages_to_parents[i].damping = 0.25
		elif method == "mk":
			self._update = self._mk_update
			# self.messages_to_parents[i].damping = 0.1
		else:
			raise ValueError(f"cannot recognize method {method}")

	def initialize_messages_to_parents(self):
		i = self._name_to_id["parent"]
		self.messages_to_parents[i] = LogisticToLogitMessage(self.parents[i], self)

	def initialize_messages_to_children(self):
		i = self._name_to_id["child"]
		self.messages_to_children[i] = LogisticToObservationMessage(self.children[i], self)

	def update_messages_to_children(self):
		# this is not the standard VMP message,
		# we rather send this message so that the posterior is the predictive posterior
		# for missing values
		# This is also fine to do, because, the message_to_factor below will
		# be fine since it removes the message in the other direction and only return the
		# observed/missing message
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		mfp = self.messages_to_parents[p_id].message_to_factor
		m, v = mfp.mean_and_variance
		# if m.isnan().any():
		# 	print(f"{self}: Mean is nan in Logistic message to child")
		# if v.isnan().any():
		# 	print(f"{self}: Variance is nan in Logistic message to child")
		proba = _ms_expit_moment(0, m.nan_to_num(), v.abs())
		self.messages_to_children[c_id].message_to_variable = Probability(proba)

	def update_messages_to_parents(self):
		self._update()

	def elbo(self, x: torch.Tensor | None = None):
		return self._elbo(x)

	def _quadratic_elbo(self, x: torch.Tensor | None = None):
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		if x is None:
			x = self.children[c_id].posterior.proba
		mfc = self.children[c_id].posterior
		mfp = self.parents[p_id].posterior
		m, v = mfp.mean_and_variance
		s = mfc.proba - 0.5
		obs = s.abs().gt(1e-10) # floating point error
		t = (m.pow(2.) + v).sqrt()
		elbo = torch.sigmoid(t).log() + s * m - t * 0.5
		elbo = torch.where(obs, elbo, torch.zeros_like(elbo))
		return elbo.sum()

	def _mk_elbo(self, x: torch.Tensor | None = None):
		# They don't really provide a way to compute the elbo, but
		# it seems this is what they are doing
		return self._quadrature_elbo(x)

	def _tilted_elbo(self, x: torch.Tensor | None = None):
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		if x is None:
			x = self.children[c_id].posterior.proba
		mfp = self.parents[p_id].posterior
		m, v = mfp.mean_and_variance
		s = 2*x - 1
		sm = s*m
		a = _tilted_fixed_point(sm, v)
		elbo = sm - 0.5*a.pow(2.)*v - _log1p_exp(sm+0.5*v*(1-2*a))
		elbo = torch.where(s.abs()==1., elbo, torch.zeros_like(elbo))
		return elbo.sum()

	def _quadrature_elbo(self, x: torch.Tensor | None = None):
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		if x is None:
			x = self.children[c_id].children.values().__iter__().__next__().values.values # N x p
		mfp = self.parents[p_id].posterior
		m, v = mfp.mean_and_variance
		s = 2.*x - 1.
		obs = s.abs().gt(1e-10) # floating point error
		sm = s*m
		elbo = sm - _gh_quadrature(sm, v, _log1p_exp)
		elbo = torch.where(obs, elbo, torch.zeros_like(elbo))
		return elbo.sum()

	def log_likelihood(self, x: torch.Tensor | None = None):
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		if x is None:
			x = self.children[c_id].posterior.proba
		mfp = self.parents[p_id].posterior
		m = mfp.mean
		s = 2*x - 1
		llk = s*m - _log1p_exp(m)
		llk = torch.where(s.abs()==1., llk, torch.zeros_like(llk))
		return llk.sum()

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

	def _adaptive_update(self):
		self._n_updates += 1
		if self._n_updates < 10:
			self._quadratic_update()
		else:
			i = self._name_to_id["parent"]
			self.messages_to_parents[i].damping = 0.1
			self._mk_update()

	def _mk_update(self):
		"""Uses Knowles & Minka (2011)"""
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		mfc = self.messages_to_children[c_id].message_to_factor
		# mfc = self.children[c_id].posterior
		# mfp = self.messages_to_parents[p_id].message_to_factor
		mfp = self.parents[p_id].posterior
		m, v = mfp.mean_and_variance
		p, mtp = mfp.precision_and_mean_times_precision
		x = mfc.proba
		x = torch.where((x-0.5).abs().lt(0.1), torch.full_like(x, 0.5), x.nan_to_num())
		b0 = _ms_expit_moment(0, m, v)
		b1 = _ms_expit_moment(1, m, v)
		# Knowles and Minka (2011, Section 5.2)
		# m1 = b1 * v.sqrt() + m * b0
		# p1 = (m1 - m * b0) * p
		# mtp1 = m * p1 + x - b0
		# Nolan and Wand (2017, Eq. 6)
		p1 = b1 * p.sqrt()
		mtp1 = m * p1 + x - b0
		p1 = torch.where((x-0.5).abs().lt(0.1), torch.zeros_like(p1), p1)
		mtp1 = torch.where((x-0.5).abs().lt(0.1), torch.zeros_like(mtp1), mtp1)
		self.messages_to_parents[p_id].message_to_variable = Normal(p1, mtp1)

	def _quadratic_update(self):
		"""Uses the quadratic bound of Jaakola and Jordan (2000)"""
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		x = x = self.children[c_id].children.values().__iter__().__next__().values.values # N x p
		mfp = self.parents[p_id].posterior
		m, v = mfp.mean_and_variance
		s = x - 0.5
		t = (m.pow(2.) + v).sqrt()
		lam = (torch.sigmoid(t) - 0.5) / t
		p1 = lam
		mtp1 = s
		p1 = torch.where(x.isnan(), torch.zeros_like(p1), p1)
		mtp1 = torch.where(x.isnan(), torch.zeros_like(mtp1), mtp1)
		self.messages_to_parents[p_id].message_to_variable = Normal(p1, mtp1)

	def _tilted_update(self):
		"""Uses the tilted bound of Saul and Jordan (1999)
		with NCVMP massages from Knowles and Minka (2011)
		"""
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		mfc = self.messages_to_children[c_id].message_to_factor
		# mfc = self.children[c_id].posterior
		# mfp = self.messages_to_parents[p_id].message_to_factor
		mfp = self.parents[p_id].posterior
		m, v = mfp.mean_and_variance
		x = mfc.proba
		a = _tilted_fixed_point(m, v)
		p1 = a * (1 - a)
		mtp1 = m * p1 + x - a
		p1 = torch.where((x-0.5).abs().lt(0.499), torch.zeros_like(p1), p1)
		mtp1 = torch.where((x-0.5).abs().lt(0.499), torch.zeros_like(mtp1), mtp1)
		self.messages_to_parents[p_id].message_to_variable = Normal(p1, mtp1)


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
		self._message_to_factor = Normal.unit_from_dimension(variable.shape)
		self._message_to_variable = Probability.unit_from_dimension(variable.shape)