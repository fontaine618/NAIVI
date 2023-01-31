import torch
import torch.distributions
from .distribution import Distribution
from .point_mass import Unit, PointMass


class Probability(Distribution):
	"""An array of probabilities"""

	_name = "Probability"

	def __init__(self, proba: torch.Tensor, check_args=None):
		super(Probability, self).__init__(dim=proba.shape, check_args=check_args)
		self.proba = proba

	@classmethod
	def unit_from_dimension(cls, dim, check_args=None):
		return Probability(torch.full(dim, 0.5), check_args=check_args)

	def __mul__(self, other):
		if type(other) is Unit:
			return self
		elif type(other) is PointMass:
			return Probability(torch.where(other.value.isnan(), self.proba, other.value))
		elif type(other) is Probability:
			p0 = self.proba
			p1 = other.proba
			p = p0 * p1
			omp = (1 - p0) * (1 - p1)
			p = p / (p + omp)
			return Probability(p)
		else:
			raise NotImplementedError(
				f"Product of Probability with {other._name} "
			    f"not implemented yet or is not meaningful."
			)

	def __pow__(self, power: float):
		pa = self.proba ** power
		ompa = (1 - self.proba) ** power
		return Probability(pa / (pa + ompa))

	def __truediv__(self, other):
		if type(other) is Unit:
			return self
		elif type(other) is PointMass:
			return other
		elif type(other) is Probability:
			p0 = self.proba
			p1 = other.proba
			p = p0 / p1
			omp = (1 - p0) / (1 - p1)
			p = p / (p + omp)
			return Probability(p)
		else:
			raise NotImplementedError(
				f"Division of Probability with {other._name} "
			    f"not implemented yet or is not meaningful."
			)

	def sample(self, n_samples: int = 1):
		return torch.distributions.Bernoulli(self.proba).sample((n_samples,))