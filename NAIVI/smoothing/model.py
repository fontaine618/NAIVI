import torch
import numpy as np
from NAIVI import MICE


class NetworkSmoothing(MICE):

    def __init__(self, K, N, p_cts, p_bin):
        self.p_cts = p_cts
        self.p_bin = p_bin
        self.N = N

    def fit(self, train, test=None, max_iter=100, **kwargs):
        # get train data
        i0, i1, A, _, X_cts, X_bin = train[:]
        if X_cts is None:
            X_cts = torch.zeros((self.N, 0))
        if X_bin is None:
            X_bin = torch.zeros((self.N, 0))
        # concatenate
        X = torch.cat([X_cts, X_bin], 1).cuda()
        missing = X.isnan()
        X_pred = None
        A_mat = torch.zeros((self.N, self.N), device=A.device)
        A_mat.index_put_((i0, i1), A.flatten())
        A_mat.index_put_((i1, i0), A.flatten())
        n_neighbors = A_mat.sum(0).reshape((-1, 1))
        n_neighbors = torch.where(n_neighbors==0., 1., n_neighbors)

        # predict
        for epoch in range(max_iter):
            if X_pred is None:  # initialize to mean
                X_mean = X.nansum(0) / (~missing).sum(0)
                X_mean = torch.vstack([X_mean for _ in range(self.N)])
                X_pred = torch.where(missing, X_mean, X)
            else:
                smooth = torch.matmul(A_mat, X_pred) / n_neighbors
                X_pred = torch.where(missing, smooth, X)
            # split
            X_cts_pred, X_bin_pred, _ = np.split(
                X_pred.cpu(), indices_or_sections=[self.p_cts, self.p_cts + self.p_bin], axis=1
            )

            # get test data
            _, _, _, _, X_test_cts, X_test_bin = test[:]
            if self.p_cts > 0:
                X_cts_pred[np.isnan(X_test_cts) == 1.0] = np.nan
                X_cts_pred = X_cts_pred.clone().detach()
            if self.p_bin > 0:
                X_bin_pred[np.isnan(X_test_bin) == 1.0] = np.nan
                X_bin_pred = X_bin_pred.clip(0.0, 1.0)
                X_bin_pred = X_bin_pred.clone().detach()

            out = self.metrics(X_test_bin, X_test_cts, X_cts_pred, X_bin_pred, epoch)
            print(f"Iteration {epoch:<3} "
                  f"Test MSE {out[('test', 'mse')]:<6.4f} "
                  f"Test AUC {out[('test', 'auc')]:<6.4f}")
        return out
