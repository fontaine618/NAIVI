import torch
import torch.distributions
from torch.distributions.multivariate_normal import _batch_mv
from .distribution import Distribution
from .point_mass import Unit, PointMass
import warnings


def _batch_mahalanobis(bP, bx):
	"""Compute x'Px batched."""
	return torch.einsum(
		"ni, nij, nj -> n",
		bx, bP, bx
	)


class Normal(Distribution):
	"""
	An array of independent Gaussian distributions.

	NB: when precision=inf, mean_times_precision stores the point mass
	"""

	_name = "Normal"

	def __init__(self, precision, mean_times_precision,
	             mean=None, variance=None, check_args=None, **kw):
		dim = mean_times_precision.shape
		super(Normal, self).__init__(dim=dim, check_args=check_args)
		precision, mean_times_precision, self._check_args(precision, mean_times_precision)
		self.precision = precision
		self.mean_times_precision = mean_times_precision
		self._mean = mean
		self._variance = variance

	@staticmethod
	def _check_args(precision, mean_times_precision):
		if Distribution._check_args:
			if precision.shape != mean_times_precision.shape:
				raise AttributeError(
					f"precision and mean_times_precision must have the same shape, "
					f"but got {precision.shape} and {mean_times_precision.shape}"
				)
			if not torch.all(precision.ge(-1.e-10)):
				raise AttributeError(
					f"precision must be nonnegative"
				)
			if precision.isnan().any():
				raise AttributeError(
					f"precision contains NaNs"
				)
			if mean_times_precision.isnan().any():
				raise AttributeError(
					f"mean_times_precision contains NaNs"
				)
		# mean_times_precision = torch.where(
		# 	precision.isinf(),
		# 	torch.zeros_like(mean_times_precision),
		# 	mean_times_precision
		# )
		return precision, mean_times_precision

	@property
	def mean(self):
		if self._mean is None:
			self._mean = torch.where(self.precision == 0., 0., self.mean_times_precision / self.precision)
			self._mean = torch.where(self.precision.isinf(), self.mean_times_precision, self._mean)
		return self._mean

	@property
	def variance(self):
		if self._variance is None:
			self._variance = 1. / self.precision
		# else:
		# 	print("variance already computed")
		return self._variance

	@property
	def precision_and_mean_times_precision(self):
		return self.precision, self.mean_times_precision

	@property
	def natural_parameters(self):
		return self.mean_times_precision, self.precision * -0.5

	@property
	def mean_and_variance(self):
		return self.mean, self.variance

	@classmethod
	def standard_from_dimension(cls, dim, check_args=None):
		precision = torch.ones(dim)
		mean_times_precision = torch.zeros(dim)
		return Normal(precision, mean_times_precision, check_args=check_args)

	@classmethod
	def unit_from_dimension(cls, dim, check_args=None):
		precision = torch.zeros(dim)
		mean_times_precision = torch.zeros(dim)
		return Normal(precision, mean_times_precision, check_args=check_args)

	@classmethod
	def point_mass(cls, point, check_args=None):
		# we allow a mix of point masses and unit
		precision = torch.full_like(point, float("inf"))
		precision = torch.where(point.isnan(), 0., precision)
		mean_times_precision = point
		mean_times_precision = torch.where(point.isnan(), 0., mean_times_precision)
		return Normal(precision, mean_times_precision, check_args=check_args)

	@classmethod
	def from_mean_and_variance(cls, mean, variance, check_args=None):
		precision = torch.where(mean.isnan(), 0., 1. / variance)
		mean_times_precision = torch.where(mean.isnan(), 0., mean * precision)
		# store point mass in mtp
		mean_times_precision = torch.where(precision.isinf(), mean, mean_times_precision)
		return Normal(precision, mean_times_precision, mean=mean, variance=variance, check_args=check_args)

	def __mul__(self, other):
		if type(other) is Normal:
			precision = self.precision + other.precision
			mean_times_precision = self.mean_times_precision + other.mean_times_precision
			mean_times_precision = torch.where(
				self.precision.isinf(),
				self.mean_times_precision,
				mean_times_precision
			)
			mean_times_precision = torch.where(
				other.precision.isinf(),
				other.mean_times_precision,
				mean_times_precision
			)
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
			# deal with the case PointMass/Gaussian=PointMass
			mean_times_precision = torch.where(self.precision.isinf(), self.mean_times_precision, mean_times_precision)
			# deal with the case PointMass/PointMass (we assume they have the same value)
			pm_pm = self.precision.isinf() * other.precision.isinf()
			precision = torch.where(pm_pm, 0., precision)
			mean_times_precision = torch.where(pm_pm, 0., mean_times_precision)
			# deal with the case Gaussian/Gaussian = Unit
			precision.clamp_(min=0.)
			mean_times_precision = torch.where(precision == 0., 0., mean_times_precision)
			return Normal(precision, mean_times_precision)
		elif type(other) is Unit:
			return self
		else:
			raise NotImplementedError(
				f"Division of Normal by {other._name} "
			    f"not implemented yet or is not meaningful."
			)

	def sample(self, n_samples: int = 1):
		return torch.normal(
			self.mean.unsqueeze(0).expand(n_samples, *self._dim),
			self.variance.unsqueeze(0).expand(n_samples, *self._dim).sqrt()
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

	def __init__(self, precision, mean_times_precision,
	             mean=None, variance=None, check_args=None, **kw):
		dim = mean_times_precision.shape
		super(MultivariateNormal, self).__init__(
			precision=precision,
			mean_times_precision=mean_times_precision,
			mean=mean,
			variance=variance,
			dim=dim,
			check_args=check_args
		)

	@staticmethod
	def _check_args(precision, mean_times_precision):
		if Distribution._check_args:
			if not (torch.isclose(
					precision, precision.transpose(-1, -2),
					rtol=1e-4, atol=1e-5)).all():
				warnings.warn(
					f"precision must be symmetric; symmetrize it"
				)
				precision = (precision + precision.transpose(-1, -2)) / 2
			if not torch.linalg.eigvalsh(precision).ge(-1.e-6).all():
				warnings.warn(
					f"precision must be positive semi-definite"
				)
		# mean_times_precision = torch.where(
		# 	precision.isinf().any(-1),
		# 	torch.zeros_like(mean_times_precision),
		# 	mean_times_precision
		# )
		return precision, mean_times_precision

	@property
	def mean(self):
		if self._mean is None:
			self._mean = _batch_mv(self.variance, self.mean_times_precision)
		# else:
		# 	print("mean already computed")
		return self._mean

	@property
	def variance(self):
		if self._variance is None:
			self._variance = torch.inverse(self.precision)
		# else:
		# 	print("covariance already computed")
		return self._variance

	@classmethod
	def standard_from_dimension(cls, dim, check_args=None):
		d = dim[-1]
		precision = torch.eye(d).expand(*dim, d)
		mean_times_precision = torch.zeros(dim)
		return MultivariateNormal(precision, mean_times_precision, check_args=check_args)

	@classmethod
	def unit_from_dimension(cls, dim, check_args=None):
		d = dim[-1]
		precision = torch.zeros(*dim, d)
		mean_times_precision = torch.zeros(dim)
		return MultivariateNormal(precision, mean_times_precision, check_args=check_args)

	@classmethod
	def point_mass(cls, point, check_args=None):
		d = point.shape[-1]
		# we allow a mix of point masses and unit
		precision = torch.full_like((*point.shape, d), float("inf"))
		precision = torch.where(point.isnan(), 0., precision)
		mean_times_precision = point
		mean_times_precision = torch.where(point.isnan(), 0., mean_times_precision)
		# make precision diagonal
		precision = torch.stack([torch.diag(p).unsqueeze(0) for p in precision], dim=0)

		return MultivariateNormal(precision, mean_times_precision, check_args=check_args)


	@classmethod
	def from_mean_and_variance(cls, mean, variance, check_args=None):
		precision = torch.inverse(variance)
		mean_times_precision = torch.matmul(precision, mean.unsqueeze(-1)).squeeze(-1)
		return MultivariateNormal(precision, mean_times_precision, mean=mean, variance=variance, check_args=check_args)

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

	def sample(self, n_samples: int = 1):
		samples = torch.distributions.multivariate_normal.MultivariateNormal(
			self.mean, self.variance
		).sample((n_samples,))
		return samples

