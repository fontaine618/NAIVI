import torch
from .distribution import Distribution


class PointMass(Distribution):

	_name = "PointMass"

	def __init__(self, value):
		dim = value.shape
		super(PointMass, self).__init__(dim=dim)
		self._value = value

	@property
	def mean(self):
		return self._value

	@property
	def variance(self):
		return torch.where(self._value.isnan(), float("Inf"), 0.)

	@property
	def precision(self):
		return torch.where(self._value.isnan(), 0., float("Inf"))

	@property
	def mean_times_precision(self):
		return torch.where(self._value.isnan(), 0., self._value)


class Unit(PointMass):
	
	_name = "Unit"
	
	def __init__(self, dim):
		value = torch.full(dim, float("NaN"))
		super(Unit, self).__init__(value)