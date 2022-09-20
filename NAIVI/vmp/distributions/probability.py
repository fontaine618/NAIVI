import torch
import torch.distributions
from .distribution import Distribution


class Probability(Distribution):
	"""An array of probabilities"""

	_name = "Probability"

	def __init__(self, proba: torch.Tensor):
		super(Probability, self).__init__(dim=proba.shape)
		self.proba = proba

	@classmethod
	def unit_from_dimension(cls, dim):
		return Probability(torch.full(dim, 0.5))