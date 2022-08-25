from ..variables import Variable
from ..distributions import Distribution


class Message:

	def __init__(self, var_from: Variable, var_to: Variable, **kw):
		self.var_from = var_from
		self.var_to = var_to
		self._dim = var_to.shape
		self.message = None # TODO: put unit message

	def update(self):
		pass
