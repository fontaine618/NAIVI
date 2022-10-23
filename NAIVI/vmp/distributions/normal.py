import torch
import torch.distributions
from torch.distributions.multivariate_normal import _batch_mv
from .distribution import Distribution
from .point_mass import Unit, PointMass


def _batch_mahalanobis(bP, bx):
	"""Compute x'Px batched."""
	return torch.einsum(
		"ni, nij, nj -> n",
		bx, bP, bx
	)


class Normal(Distribution):
	"""
	An array of independent Gaussian distributions.
	"""

	_name = "Normal"

	def __init__(self, precision, mean_times_precision, **kw):
		dim = mean_times_precision.shape
		super(Normal, self).__init__(dim=dim)
		self._check_args(precision, mean_times_precision)
		self.precision = precision
		self.mean_times_precision = mean_times_precision

	@staticmethod
	def _check_args(precision, mean_times_precision):
		if precision.shape != mean_times_precision.shape:
			raise AttributeError(
				f"precision and mean_times_precision must have the same shape, "
				f"but got {precision.shape} and {mean_times_precision.shape}"
			)
		if not torch.all(precision.ge(-1.e-10)):
			raise AttributeError(
				f"precision must be nonnegative"
			)

	@property
	def mean(self):
		return torch.where(self.precision == 0., 0., self.mean_times_precision / self.precision)

	@property
	def variance(self):
		return 1. / self.precision

	@property
	def precision_and_mean_times_precision(self):
		return self.precision, self.mean_times_precision

	@property
	def natural_parameters(self):
		return self.mean_times_precision, self.precision * -0.5

	@property
	def mean_and_variance(self):
		var = self.variance
		mean = torch.where(self.precision == 0., 0., self.mean_times_precision * var)
		return mean, var

	@classmethod
	def standard_from_dimension(cls, dim):
		precision = torch.ones(dim)
		mean_times_precision = torch.zeros(dim)
		return Normal(precision, mean_times_precision)

	@classmethod
	def unit_from_dimension(cls, dim):
		precision = torch.zeros(dim)
		mean_times_precision = torch.zeros(dim)
		return Normal(precision, mean_times_precision)

	@classmethod
	def from_mean_and_variance(cls, mean, variance):
		precision = torch.where(mean.isnan(), 0., 1. / variance)
		mean_times_precision = torch.where(mean.isnan(), 0., mean * precision)
		return Normal(precision, mean_times_precision)

	def __mul__(self, other):
		if type(other) is Normal:
			precision = self.precision + other.precision
			mean_times_precision = self.mean_times_precision + other.mean_times_precision
			return Normal(precision, mean_times_precision)
		elif type(other) is Unit:
			return self
		elif type(other) is PointMass:
			return other
		else:
			raise NotImplementedError(
				f"Product of Normal with {other._name} "
			    f"not implemented yet or is not meaningful."
			)

	def __pow__(self, power: float):
		return Normal(
			self.precision * power,
			self.mean_times_precision * power
		)

	def __truediv__(self, other: Distribution):
		if type(other) is Normal:
			precision = self.precision - other.precision
			mean_times_precision = self.mean_times_precision - other.mean_times_precision
			return Normal(precision, mean_times_precision)
		elif type(other) is Unit:
			return self
		else:
			raise NotImplementedError(
				f"Product of Normal with {other._name} "
			    f"not implemented yet or is not meaningful."
			)

	def unsqueeze(self, dim):
		return Normal(
			precision=self.precision.unsqueeze(dim),
			mean_times_precision=self.mean_times_precision.unsqueeze(dim)
		)

	def expand(self, *sizes):
		return Normal(
			precision=self.precision.expand(*sizes),
			mean_times_precision=self.mean_times_precision.expand(*sizes)
		)

	def prod(self, dim):
		p = self.precision.sum(dim)
		mtp = self.mean_times_precision.sum(dim)
		return Normal(p, mtp)

	def index_select(self, dim, index):
		return Normal(
			precision=self.precision.index_select(dim, index),
			mean_times_precision=self.mean_times_precision.index_select(dim, index)
		)

	def index_sum(self, dim, index, max_index):
		zeros = torch.zeros(max_index, *self.precision.shape[1:])
		precision = zeros.index_add(dim, index, self.precision)
		mean_times_precision = zeros.index_add(dim, index, self.mean_times_precision)
		return Normal(
			precision=precision,
			mean_times_precision=mean_times_precision
		)

	def clone(self):
		return Normal(
			precision=self.precision.clone(),
			mean_times_precision=self.mean_times_precision.clone()
		)


class MultivariateNormal(Normal):

	_name = "MultivariateNormal"

	def __init__(self, precision, mean_times_precision, **kw):
		dim = mean_times_precision.shape
		super(MultivariateNormal, self).__init__(
			precision=precision,
			mean_times_precision=mean_times_precision,
			dim=dim
		)

	@staticmethod
	def _check_args(precision, mean_times_precision):
		if not (torch.isclose(precision, precision.transpose(-1, -2))).all():
			raise ValueError(
				f"precision must be symmetric"
			)
		if not torch.linalg.eigvalsh(precision).ge(-1.e-6).all():
			print(
				f"precision must be positive semi-definite"
			)

	@property
	def mean(self):
		return _batch_mv(self.variance, self.mean_times_precision)

	@property
	def variance(self):
		return torch.inverse(self.precision)

	@property
	def mean_and_variance(self):
		var = self.variance
		mean = _batch_mv(var, self.mean_times_precision)
		return mean, var

	@classmethod
	def standard_from_dimension(cls, dim):
		d = dim[-1]
		precision = torch.eye(d).expand(*dim, d)
		mean_times_precision = torch.zeros(dim)
		return MultivariateNormal(precision, mean_times_precision)

	@classmethod
	def unit_from_dimension(cls, dim):
		d = dim[-1]
		precision = torch.zeros(*dim, d)
		mean_times_precision = torch.zeros(dim)
		return MultivariateNormal(precision, mean_times_precision)

	@classmethod
	def from_mean_and_variance(cls, mean, variance):
		precision = torch.inverse(variance)
		mean_times_precision = torch.matmul(precision, mean.unsqueeze(-1)).squeeze(-1)
		return MultivariateNormal(precision, mean_times_precision)

	def __mul__(self, other):
		if type(other) is MultivariateNormal:
			precision = self.precision + other.precision
			mean_times_precision = self.mean_times_precision + other.mean_times_precision
			return MultivariateNormal(precision, mean_times_precision)
		elif type(other) is Unit:
			return self
		elif type(other) is PointMass:
			return other
		else:
			raise NotImplementedError(
				f"Product of MultivariateNormal with {other._name} "
			    f"not implemented yet or is not meaningful."
			)

	def __pow__(self, power: float):
		return MultivariateNormal(
			self.precision * power,
			self.mean_times_precision * power
		)

	def __truediv__(self, other):
		if type(other) is MultivariateNormal:
			precision = self.precision - other.precision
			mean_times_precision = self.mean_times_precision - other.mean_times_precision
			return MultivariateNormal(precision, mean_times_precision)
		elif type(other) is Unit:
			return self
		else:
			raise NotImplementedError(
				f"Product of MultivariateNormal with {other._name} "
			    f"not implemented yet or is not meaningful."
			)

	def unsqueeze(self, dim):
		return MultivariateNormal(
			precision=self.precision.unsqueeze(dim),
			mean_times_precision=self.mean_times_precision.unsqueeze(dim)
		)

	def expand(self, *sizes):
		return MultivariateNormal(
			precision=self.precision.expand(*(*sizes, -1)),
			mean_times_precision=self.mean_times_precision.expand(*sizes)
		)

	def prod(self, dim):
		p = self.precision.sum(dim)
		mtp = self.mean_times_precision.sum(dim)
		return MultivariateNormal(p, mtp)

	def index_select(self, dim, index):
		return MultivariateNormal(
			precision=self.precision.index_select(dim, index),
			mean_times_precision=self.mean_times_precision.index_select(dim, index)
		)

	def index_sum(self, dim, index, max_index):
		zeros = torch.zeros(max_index, *self.precision.shape[1:])
		precision = zeros.index_add(dim, index, self.precision)
		zeros = torch.zeros(max_index, *self.mean_times_precision.shape[1:])
		mean_times_precision = zeros.index_add(dim, index, self.mean_times_precision)
		return MultivariateNormal(
			precision=precision,
			mean_times_precision=mean_times_precision
		)

	def clone(self):
		return MultivariateNormal(
			precision=self.precision.clone(),
			mean_times_precision=self.mean_times_precision.clone()
		)
