import torch
import numpy as np
from torch.nn.functional import mse_loss


def invariant_distance(y_true, y_pred):
    with torch.no_grad():
        ip_true = torch.matmul(y_true, y_true.transpose(0, 1))
        ip_pred = torch.matmul(y_pred, y_pred.transpose(0, 1))
        diff = ip_true - ip_pred
        num = (diff ** 2).sum()
        denum = (ip_true ** 2).sum()
    return (num / denum).item()


def projection_distance(y_true, y_pred):
    try:
        with torch.no_grad():
            u, _, _ = torch.svd(y_true)
            proj_true = torch.matmul(u, u.transpose(0, 1))
            u, _, _ = torch.svd(y_pred)
            proj_pred = torch.matmul(u, u.transpose(0, 1))
        return ((proj_pred - proj_true) ** 2).sum().item()
    except Exception:
        return np.nan


def proba_distance(Z, a, ZZ, aa):
    P = compute_prob(Z, a)
    PP = compute_prob(ZZ, aa)
    return mse_loss(P, PP).item()


def compute_prob(Z, a):
    with torch.no_grad():
        Theta = Z @ Z.T + a + a.T
        return torch.sigmoid(Theta)