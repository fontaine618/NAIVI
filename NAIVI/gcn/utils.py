import torch


def normalize(mx):
    """Row-normalize matrix"""
    rsum = mx.sum(1)
    inv = torch.where(rsum == 0, torch.zeros_like(rsum), 1.0 / rsum)
    inv = inv.reshape(-1)
    inv = torch.diag(inv)
    mx = inv @ mx
    return mx
