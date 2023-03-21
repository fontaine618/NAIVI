from __future__ import annotations
import torch
from .factor import Factor
from ..distributions.normal import Normal, MultivariateNormal
from ..messages import Message

from typing import TYPE_CHECKING, Dict
if TYPE_CHECKING:
	from ..variables import Variable
	from ..messages import Message


class Affine(Factor):
	"""child = bias + weight'parent"""

	_name = "Affine"
	_deterministic = True

	def __init__(self, dim_in: int, dim_out: int, parent: Variable):
		self.parameters: Dict[str, torch.nn.Parameter] = dict()
		self.parameters["bias"] = torch.nn.Parameter(torch.randn(dim_out))
		self.parameters["weights"] = torch.nn.Parameter(torch.randn((dim_in, dim_out)))
		super(Affine, self).__init__(parent=parent)

	def initialize_messages_to_parents(self):
		i = self._name_to_id["parent"]
		self.messages_to_parents[i] = AffineToParentMessage(self.parents[i], self)

	def initialize_messages_to_children(self):
		i = self._name_to_id["child"]
		self.messages_to_children[i] = AffineToChildMessage(self.children[i], self)

	def update_messages_to_children(self):
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfp = self.messages_to_parents[p].message_to_factor  # N x p x K
		b = self.parameters["bias"].data  # p
		w = self.parameters["weights"].data  # K x p
		m, v = mfp.mean_and_variance  # N x p x k, N x p x K x K
		mean = torch.einsum("kj, ijk -> ij", w, m)  # N x p
		mean += b.reshape((1, -1))  # 1 x p
		variance = torch.einsum("kj, ijkl, lj -> ij", w, v, w)  # N x p
		self.messages_to_children[c].message_to_variable = Normal.from_mean_and_variance(mean, variance)

	def update_messages_to_parents(self):
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfc = self.messages_to_children[c].message_to_factor  # N x p
		mc = mfc.mean  # N x p, N x p
		pc = mfc.precision  # N x p
		b = self.parameters["bias"].data  # p
		w = self.parameters["weights"].data  # K x p
		mmb = mc - b.reshape((1, -1))  # N x p
		prec = torch.einsum("kj, lj, ij -> ijkl", w, w, pc)  # N x p x K x K
		mtp = torch.einsum("ij, kj, ij -> ijk", mmb, w, pc)  # N x p x K
		self.messages_to_parents[p].message_to_variable = MultivariateNormal(prec, mtp)

	def update_messages_from_parents(self):
		# this is a deterministic factor, so we should divide
		for i, parent in self.parents.items():
			mtp = self.messages_to_parents[i]._message_to_variable  # we need all messages here
			post = parent.posterior.unsqueeze(1)  #.expand(-1, mtp.shape[1], -1)  # N x p x K
			self.messages_to_parents[i].message_to_factor = post / mtp

	def update_parameters(self):
		# TODO: check this
		self._update_bias()
		self._update_weights()

	def _update_bias(self):
		# NB: missing values will have precision 0 and therefore will not count towards any sums
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfc = self.messages_to_children[c].message_to_factor
		mfp = self.messages_to_parents[p].message_to_factor
		mp = mfp.mean  # N x p x K, N x p x K x K
		pc, mtpc = mfc.precision_and_mean_times_precision  # N x p, N x p
		B = self.parameters["weights"]  # K x p
		WR = mtpc.sum(0) - torch.einsum("ijk, kj, ij -> j", mp, B, pc)
		W = pc.sum(0) # N x p x K

		# check that it is exactly equivalent for Gaussian model
		# mean_c = mfc.mean
		# Btmean_p = torch.einsum("ijk, kj -> ij", mp, B)
		# diff = mean_c - Btmean_p
		# diff = torch.where(pc < 1e-10, torch.full_like(diff, float("nan")), diff)
		# diff.nanmean(0)

		self.parameters["bias"].data = WR / W

	def _update_weights(self):
		# NB: missing values will have precision 0 and therefore will not count towards any sums
		p = self._name_to_id["parent"]
		c = self._name_to_id["child"]
		mfc = self.messages_to_children[c].message_to_factor
		mfp = self.messages_to_parents[p].message_to_factor
		mp, vp = mfp.mean_and_variance  # N x p x K, N x p x K x K
		pc, mtpc = mfc.precision_and_mean_times_precision  # N x p, N x p
		B0 = self.parameters["bias"].data
		# X'X
		Vpmmt = vp + torch.einsum("ijk, ijl -> ijkl", mp, mp)
		SpVpmmt = torch.einsum("ij, ijkl -> jkl", pc, Vpmmt)
		SpVpmmt += torch.eye(SpVpmmt.shape[0]) * 0.01
		# X'Y
		pr = mtpc - B0.reshape((1, -1)) * pc
		Sprm = torch.einsum("ij, ijk -> kj", pr, mp)
		self.parameters["weights"].data = torch.linalg.solve(SpVpmmt, Sprm.T).T

	def forward(self, **kwargs):
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		sp = self.parents[p_id].samples # B x N x p
		b = self.parameters["bias"].data  # p
		w = self.parameters["weights"].data  # K x p
		sc = torch.einsum("kj, bik -> bij", w, sp)  # B x N x p
		sc += b.reshape((1, 1, -1))  # 1 x 1 x p
		self.children[c_id].samples = sc

	def weights_entropy(self):
		p_id = self._name_to_id["parent"]
		c_id = self._name_to_id["child"]
		mfc = self.messages_to_children[c_id].message_to_factor
		mfp = self.messages_to_parents[p_id].message_to_factor
		mp, vp = mfp.mean_and_variance  # N x p x K, N x p x K x K
		pc, mtpc = mfc.precision_and_mean_times_precision  # N x p, N x p
		B0 = self.parameters["bias"].data
		# compute message parameters
		mtpc_mB0 = mtpc - B0.reshape((1, -1)) * pc # N x p
		mtp = torch.einsum("ij, ik->jk", mtpc_mB0, pc) # K x p
		var_p_mmt = vp + mp.unsqueeze(-1) * mp.unsqueeze(-2) # N x p x K x K
		p = torch.einsum("ij, ijkl->jkl", pc, var_p_mmt) # K x p
		# compute entropy
		entropy = 0.5 * (torch.logdet(p * 2 * torch.pi) + p.shape[-1])
		return entropy.sum()


class AffineToParentMessage(Message):

	_name = "AffineToParentMessage"

	def __init__(self, variable: Variable, factor: Affine):
		super(AffineToParentMessage, self).__init__(
			variable=variable,
			factor=factor
		)
		dim = variable.shape
		dim_parallel = factor.parameters["weights"].shape[1]
		self._dim = torch.Size([dim[0], dim_parallel, *dim[1:]])
		self._message_to_factor = MultivariateNormal.unit_from_dimension(self._dim)
		self._message_to_variable = MultivariateNormal.unit_from_dimension(self._dim)

	def _get_message_to_variable(self):
		return self._message_to_variable.prod(1)

	def _set_message_to_variable(self, msg):
		"""Stores the new message and updates the posterior of the variable."""
		# print(f"Update message from {repr(self.factor)} to {self.variable}")
		prev_msg = self.message_to_variable
		self._message_to_variable = msg
		self.variable.update(prev_msg, msg.prod(1))

	def _del_message_to_variable(self):
		del self._message_to_variable

	message_to_variable = property(
		_get_message_to_variable,
		_set_message_to_variable,
		_del_message_to_variable
	)


class AffineToChildMessage(Message):

	_name = "AffineToChildMessage"

	def __init__(self, variable: Variable, factor: Affine):
		super(AffineToChildMessage, self).__init__(variable, factor)
		self._message_to_factor = Normal.unit_from_dimension(variable.shape)
		self._message_to_variable = Normal.unit_from_dimension(variable.shape)