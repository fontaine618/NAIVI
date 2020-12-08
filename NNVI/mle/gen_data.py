import torch
import numpy as np

def generate_dataset(
        N, K, p_cts, p_bin, var_adj=1., var_cov=1., missing_rate=0.2, alpha_mean=-1.,
        seed=1, link_model="Logistic", bin_model="Logistic"
):
    torch.manual_seed(seed)
    # parameters
    p = p_cts + p_bin
    B = torch.randn((K, p))
    B = B / torch.norm(B, dim=0, keepdim=True)
    B0 = torch.zeros((1, p))

    # produce all link indices
    i = [[(i, j) for j in range(i + 1, N)] for i in range(0, N - 1)]
    i = torch.tensor(sum(i, []))
    i0 = i[:, 0]
    i1 = i[:, 1]

    # generate latent
    Z = torch.randn((N, K))
    alpha = torch.randn((N, 1)) * 0.5 + alpha_mean

    # inner product model
    A_logit = alpha[i0] + alpha[i1] + torch.sum(Z[i0, :] * Z[i1, :], 1, keepdim=True)
    A_proba = torch.sigmoid(A_logit)
    A = (torch.rand_like(A_proba) < A_proba).float()

    # mean model
    X_mean = B0 + torch.matmul(Z, B)
    mean_cts, logit_bin = torch.split(X_mean, (p_cts, p_bin), 1)
    X_cts_all = (mean_cts + torch.randn_like(mean_cts) * torch.sqrt(torch.tensor(var_cov))).double()
    X_bin_all = (torch.rand_like(logit_bin) < torch.sigmoid(logit_bin)).double()

    # insert missing values
    mask_cts = torch.rand_like(X_cts_all) < missing_rate
    X_cts = torch.where(mask_cts, np.nan, X_cts_all)
    X_cts_missing = torch.where(~mask_cts, np.nan, X_cts_all)
    if p_cts == 0:
        X_cts = None
        X_cts_missing = None

    mask_bin = torch.rand_like(X_bin_all) < missing_rate
    X_bin = torch.where(mask_bin, np.nan, X_bin_all)
    X_bin_missing = torch.where(~mask_bin, np.nan, X_bin_all)
    if p_bin == 0:
        X_bin = None
        X_bin_missing = None
    return Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0