import torch


class Distribution:

	_name = "Distribution"
	_check_args = __debug__

	@staticmethod
	def set_default_check_args(value: bool) -> None:
		"""
		Sets whether validation is enabled or disabled.
		The default behavior mimics Python's ``assert`` statement: validation
		is on by default, but is disabled if Python is run in optimized mode
		(via ``python -O``). Validation may be expensive, so you may want to
		disable it once a model is working.
		Args:
			value (bool): Whether to enable validation.
		"""
		if value not in [True, False]:
			raise ValueError
		Distribution._check_args = value

	def __init__(self, dim=None, check_args=None, **kw):
		self._dim = torch.Size(dim)
		if check_args is not None:
			self._check_args = check_args

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

	def __pow__(self, power: float):
		return self

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
	def precision_and_mean_times_precision(self):
		pass

	@property
	def mean_and_variance(self):
		return self.mean, self.variance

	def prod(self, dim):
		pass

	def unsqueeze(self, dim):
		pass

	def expand(self, *sizes):
		pass

	def index_select(self, dim, index):
		pass

	def index_sum(self, dim, index, max_index):
		pass

	def clone(self):
		pass

	def sample(self, n_samples: int = 1):
		pass
