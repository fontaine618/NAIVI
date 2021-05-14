import torch
import numpy as np


def to_adj_matrix(i0, i1, A, N=None):
	if N is None:
		N = max(i0.max(), i1.max()) + 1
	A_mat = torch.zeros((N, N), device=A.device)
	A_mat.index_put_((i0, i1), A.flatten())
	A_mat.index_put_((i1, i0), A.flatten())
	return A_mat


def usvt(A_mat, tau=None):
	u, s, v = torch.svd(A_mat)
	N = A_mat.shape[0]
	if tau is None:
		tau = 2.001 * np.sqrt(N * A_mat.mean().item())
	c = (s >= tau).sum().item()
	return u[:, :c] @ torch.diag(s[:c]) @ v[:, :c].T


def usvt_logit(A_mat, tau=None):
	P_hat = usvt(A_mat, tau)
	P_hat.clamp_(0.01, 0.99)
	return torch.logit(0.5 * (P_hat + P_hat.t()))


def project_rank_one_with_residuals(Theta):
	alpha_hat, approximation = flat_approximation(Theta)
	residuals = Theta - approximation
	return alpha_hat, residuals


def flat_approximation(Theta):
	N = Theta.shape[0]
	a = torch.nn.Parameter(torch.zeros((N, 1), device=Theta.device), requires_grad=True)
	opt = torch.optim.Adam([a], lr=0.01)
	for i in range(1000):
		opt.zero_grad()
		fitted = a + a.T
		loss = ((Theta - fitted)**2).mean()
		loss.backward()
		opt.step()
	return a.data, fitted.data


def doubly_center(residuals):
	mean = residuals.mean(0, keepdims=True)
	residuals = residuals - mean
	mean = residuals.mean(1, keepdims=True)
	return residuals - mean


def project_symmetric_pd(residuals):
	symmetric = 0.5 * (residuals + residuals.t())
	evals, evecs = torch.symeig(symmetric, eigenvectors=True)
	evals = torch.where(evals < 0., 0., evals)
	return evecs @ torch.diag(evals) @ evecs.t()


def initialize_latent_variables(i0, i1, A, K, tau=None):
	A_mat = to_adj_matrix(i0, i1, A)
	Theta = usvt_logit(A_mat, tau)
	alpha_hat, residuals = project_rank_one_with_residuals(Theta)
	residuals = doubly_center(residuals)
	proj = project_symmetric_pd(residuals)
	u, s, _ = torch.svd(proj)
	Z_hat = u[:, :K] @ torch.diag(torch.sqrt(s[:K]))
	return alpha_hat, Z_hat