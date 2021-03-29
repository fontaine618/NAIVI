import torch
import numpy as np
from torch.utils.data.dataloader import Dataset


class JointDataset(Dataset):

    def __init__(self, i0, i1, A, X_cts=None, X_bin=None,
                 cuda=True, return_missingness=False, test=False):
        super().__init__()
        self.i0 = i0.cuda()
        self.i1 = i1.cuda()
        self.A = A.cuda()
        self.X_cts = X_cts
        self.X_bin = X_bin
        self.return_missingness = return_missingness
        self.N = torch.cat([i0, i1]).unique().shape[0]
        self.p_bin_no_missingness = X_bin.shape[1] if X_bin is not None else 0
        if X_cts is None:
            X_cts = torch.zeros((self.N, 0))
        if X_bin is None:
            X_bin = torch.zeros((self.N, 0))
        if return_missingness:
            M_cts = torch.isnan(X_cts).double()
            M_bin = torch.isnan(X_bin).double()
            if test:
                M_cts *= np.nan
                M_bin *= np.nan
            X_bin = torch.cat([X_bin, M_cts, M_bin], 1)
            self.X_bin = X_bin
        self.p_cts = X_cts.size(1)
        self.p_bin = X_bin.size(1)
        if self.X_cts is not None:
            self.X_cts.cuda()
        if self.X_bin is not None:
            self.X_bin.cuda()

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        i0 = self.i0[i]
        i1 = self.i1[i]
        A = self.A[i]
        j = torch.cat([i0, i1], 0).unique()
        X_cts = None
        X_bin = None
        if self.X_cts is not None:
            X_cts = self.X_cts[j, :]
        if self.X_bin is not None:
            X_bin = self.X_bin[j, :]
        return i0, i1, A, j, X_cts, X_bin

    def cv_fold(self, fold=0, n_folds=5, seed=0):
        torch.manual_seed(seed)
        X_cts = self.X_cts if self.X_cts is not None else torch.ones((self.N, 0))
        X_bin = self.X_bin[:, 0:self.p_bin_no_missingness] if self.X_bin is not None else torch.ones((self.N, 0))
        fold_cts = torch.randint(0, n_folds, X_cts.shape)
        fold_bin = torch.randint(0, n_folds, X_bin.shape)
        X_cts_fold = torch.where(fold_cts == fold, X_cts, np.nan)
        X_cts = torch.where(fold_cts == fold, np.nan, X_cts)
        X_bin_fold = torch.where(fold_bin == fold, X_bin, np.nan)
        X_bin = torch.where(fold_bin == fold, np.nan, X_bin)
        train = JointDataset(self.i0, self.i1, self.A, X_cts, X_bin,
                             return_missingness=self.return_missingness)
        test = JointDataset(self.i0, self.i1, self.A, X_cts_fold, X_bin_fold,
                            return_missingness=self.return_missingness)
        # patch train: set M=NaN for removed entries
        M = train.X_bin[:, self.p_bin_no_missingness:]
        if M.shape[1]>0:
            which = torch.isnan(torch.cat([X_cts_fold, X_bin_fold], 1))
            M = torch.where(which, M, np.nan)
        train.X_bin[:, self.p_bin_no_missingness:] = M
        # patch test
        test.X_bin[:, self.p_bin_no_missingness:] = np.nan
        return train, test

    def cv_folds(self, n_folds=5, seed=0):
        return {
            fold: self.cv_fold(fold, n_folds, seed) for fold in range(n_folds)
        }