import torch
from .distribution import Distribution


class PointMass(Distribution):

	_name = "PointMass"

	def __init__(self, value):
		dim = value.shape
		super(PointMass, self).__init__(dim=dim)
		self._value = value

	@property
	def value(self):
		return self._value

	@property
	def mean(self):
		return torch.where(self._value.isnan(), 0., self._value)

	@property
	def variance(self):
		return torch.where(self._value.isnan(), float("Inf"), 0.)

	@property
	def precision(self):
		return torch.where(self._value.isnan(), 0., float("Inf"))

	@property
	def mean_times_precision(self):
		return torch.where(self._value.isnan(), 0., self._value)

	def __truediv__(self, other):
		if isinstance(other, PointMass):
			if (torch.nan_to_num(self._value, nan=-1.) != torch.nan_to_num(other._value, nan=-1.)).any():
				raise ValueError("cannot dive PointMass if not exactly equal")
			return Unit(self._dim)
		# all(?) other cases
		return self

	def sample(self, n_samples: int = 1):
		return self._value.expand(n_samples, *self._dim)


class Unit(PointMass):
	
	_name = "Unit"
	
	def __init__(self, dim):
		value = torch.full(dim, float("NaN"))
		super(Unit, self).__init__(value)