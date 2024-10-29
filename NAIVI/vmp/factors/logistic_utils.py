from __future__ import annotations

import math
import numpy as np

import torch.distributions

_p = torch.tensor([
	0.003246343272134,
	0.051517477033972,
	0.195077912673858,
	0.315569823632818,
	0.274149576158423,
	0.131076880695470,
	0.027912418727972,
	0.001449567805354
])
_s = torch.tensor([
	1.365340806296348,
	1.059523971016916,
	0.830791313765644,
	0.650732166639391,
	0.508135425366489,
	0.396313345166341,
	0.308904252267995,
	0.238212616409306
])

_logHalfUlpPrev1 = math.log(0.5 * math.ulp(1.0))
INV_SQRT_PI = 1. / math.sqrt(math.pi)
_gh_nodes, _gh_weights = np.polynomial.hermite.hermgauss(101)
_gh_nodes = torch.Tensor(_gh_nodes)
_gh_weights = torch.Tensor(_gh_weights)


def _gh_quadrature(
		mean: torch.Tensor,
		variance: torch.Tensor,
		function: callable[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
	"""Approximates E{f(X)}, X~N(mean, variance), using Gauss-Hermite quadrature.

	Inspired from GPyTorch (https://docs.gpytorch.ai/en/stable/_modules/gpytorch/utils/quadrature.html)"""
	n_dims_to_pad = len(mean.shape)
	device = mean.device
	locations = _gh_nodes.to(device).view(*([1] * n_dims_to_pad), -1)
	weights = _gh_weights.to(device).view(*([1] * n_dims_to_pad), -1) * INV_SQRT_PI
	shifted_nodes = mean.unsqueeze(-1) + locations * (variance * 2.).sqrt().unsqueeze(-1)
	func_evals = function(shifted_nodes)
	return (func_evals * weights).sum(-1)


def _log1p_exp(x: torch.Tensor) -> torch.Tensor:
	return torch.where(x > -_logHalfUlpPrev1, x, torch.log1p(torch.exp(x)))


def _tilted_fixed_point(mean, variance, max_iter=20):
	variance.clamp_max_(1e6)
	a = torch.full_like(mean, 0.5)
	for _ in range(max_iter):
		a = torch.sigmoid(mean - (1 - 2 * a) * variance / 2)
	return a


def _ms_expit_moment(degree: int, mean: torch.Tensor, variance: torch.Tensor):
	n = len(mean.shape)
	mean = mean.unsqueeze(-1)
	variance = variance.unsqueeze(-1).clamp_max(1e6)
	device = mean.device
	p = _p.to(device)
	s = _s.to(device)
	for _ in range(n):
		p = p.unsqueeze(0)
		s = s.unsqueeze(0)
	sqrt = (1 + variance * s.pow(2)).sqrt()
	arg = mean * s / sqrt
	if degree == 0:
		f1 = p
		f2 = torch.distributions.normal.Normal(0., 1.).cdf(arg)
		return (f1 * f2).sum(-1)
	elif degree == 1:
		f1 = p * s / sqrt
		f2 = torch.distributions.normal.Normal(0., 1.).log_prob(arg).exp()
		return (f1 * f2).sum(-1) * variance.sqrt().squeeze(-1)
	else:
		raise NotImplementedError("only degree 0 or 1 implemented")