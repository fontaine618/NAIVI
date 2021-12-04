import torch
import numpy as np

torch.set_default_dtype(torch.float64)


def generate_dataset(
        N, K, p_cts, p_bin, var_cov=1., alpha_mean=-1.,
        mnar_sparsity=0.5, missing_mean=-1.0, seed=1,
        adjacency_noise=0.
):
    torch.manual_seed(seed)
    # parameters
    p = p_cts + p_bin
    B = torch.randn((K, p))
    B = B / torch.abs(B)
    B0 = torch.zeros((1, p))
    C = B.detach().clone()
    which = torch.rand(p) > mnar_sparsity
    C *= which.reshape((1, -1))
    C0 = torch.ones((1, p)) * missing_mean

    # produce all link indices
    i = torch.tril_indices(N, N, -1)
    i0 = i[0, :]
    i1 = i[1, :]

    # generate latent
    Z = torch.randn((N, K))
    Z -= Z.mean(0)
    alpha = torch.randn((N, 1)) * 0.5 + alpha_mean

    # inner product model
    A_logit = alpha[i0] + alpha[i1] + torch.sum(Z[i0, :] * Z[i1, :], 1, keepdim=True)
    A_logit += torch.randn_like(A_logit) * torch.sqrt(adjacency_noise)
    A_proba = torch.sigmoid(A_logit)
    A = (torch.rand_like(A_proba) < A_proba).double()

    # mean model
    BC0 = torch.cat([B0, C0], 1)
    BC = torch.cat([B, C], 1)
    XM_mean = BC0 + torch.matmul(Z, BC)
    mean_cts, logit_bin, logit_missing_cts, logit_missing_bin = \
        torch.split(XM_mean, (p_cts, p_bin, p_cts, p_bin), 1)
    X_cts_all = (mean_cts + torch.randn_like(mean_cts) * torch.sqrt(torch.tensor(var_cov))).double()
    X_bin_all = (torch.rand_like(logit_bin) < torch.sigmoid(logit_bin)).double()
    M_cts = (torch.rand_like(logit_missing_cts) < torch.sigmoid(logit_missing_cts)).double()
    M_bin = (torch.rand_like(logit_missing_bin) < torch.sigmoid(logit_missing_bin)).double()

    # insert missing values
    mask_cts = M_cts > 0.5
    X_cts = torch.where(mask_cts, np.nan, X_cts_all)
    X_cts_missing = torch.where(~mask_cts, np.nan, X_cts_all)
    if p_cts == 0:
        X_cts = None
        X_cts_missing = None

    mask_bin = M_bin > 0.5
    X_bin = torch.where(mask_bin, np.nan, X_bin_all)
    X_bin_missing = torch.where(~mask_bin, np.nan, X_bin_all)
    if p_bin == 0:
        X_bin = None
        X_bin_missing = None
    return Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0