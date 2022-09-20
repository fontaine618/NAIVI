import torch


class Distribution:

	_name = "Distribution"

	def __init__(self, dim=None, **kw):
		self._dim = torch.Size(dim)

	def __repr__(self):
		dim_str = ""
		if self._dim is not None:
			dim_str = ", ".join([str(dim) for dim in self._dim])
		return f"{self._name}(dim=[{dim_str}])"

	def __str__(self):
		return repr(self)

	def __mul__(self, other):
		pass

	__rmul__ = __mul__

	def __truediv__(self, other):
		pass

	def __getitem__(self, tup):
		pass

	@property
	def shape(self):
		return self._dim

	def density(self, value) -> torch.Tensor:
		return self.log_density(value).exp()

	def log_density(self, value) -> torch.Tensor:
		pass

	@property
	def entropy(self) -> torch.Tensor:
		return torch.zeros(self._dim)

	def cross_entropy(self, other) -> torch.Tensor:
		pass

	@property
	def mean(self):
		pass

	@property
	def variance(self):
		pass

	@property
	def mean_and_variance(self):
		return self.mean, self.variance
