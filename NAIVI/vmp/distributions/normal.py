import torch
import torch.distributions
from torch.distributions.multivariate_normal import _batch_mv
from .distribution import Distribution, DistributionArray


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
		dim = precision.shape
		super(Normal, self).__init__(dim=dim)
		self.precision = precision
		self.mean_times_precision = mean_times_precision

	@property
	def mean(self):
		return self.mean_times_precision / self.precision

	@property
	def variance(self):
		return 1. / self.precision

	@property
	def precision_and_mean_times_precision(self):
		return self.precision, self.mean_times_precision

	@property
	def mean_and_variance(self):
		var = self.variance
		mean = self.mean_times_precision * var
		return mean, var

	@classmethod
	def standard_from_dimension(cls, dim):
		precision = torch.ones(dim)
		mean_times_precision = torch.zeros(dim)
		return Normal(precision, mean_times_precision)

	@classmethod
	def from_mean_and_variance(cls, mean, variance):
		precision = 1. / variance
		mean_times_precision = mean * precision
		return Normal(precision, mean_times_precision)

	def __mul__(self, other):
		if isinstance(other, Normal):
			precision = self.precision + other.precision
			mean_times_precision = self.mean_times_precision + other.mean_times_precision
			return Normal(precision, mean_times_precision)
		else:
			raise ValueError(f"Product of Normal with {other.__class__.__name__} "
			                 f"not implemented yet or is not meaningful.")

	def __truediv__(self, other):
		if isinstance(other, Normal):
			precision = self.precision - other.precision
			mean_times_precision = self.mean_times_precision - other.mean_times_precision
			return Normal(precision, mean_times_precision)
		else:
			raise ValueError(f"Product of Normal with {other.__class__.__name__} "
			                 f"not implemented yet or is not meaningful.")

	__rmul__ = __mul__


class MultivariateNormal(Normal, DistributionArray):

	_name = "MultivariateNormal"

	def __init__(self, precision, mean_times_precision, **kw):
		dim = mean_times_precision.shape
		super(MultivariateNormal, self).__init__(
			precision=precision,
			mean_times_precision=mean_times_precision,
			dim=dim
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
	def from_mean_and_variance(cls, mean, variance):
		precision = torch.inverse(variance)
		mean_times_precision = torch.matmul(precision, mean.unsqueeze(-1)).squeeze(-1)
		return MultivariateNormal(precision, mean_times_precision)
