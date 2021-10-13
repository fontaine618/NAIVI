import torch
import numpy as np
from torch.utils.data.dataloader import Dataset


class JointDataset(Dataset):

    def __init__(self, i0, i1, A, X_cts=None, X_bin=None,
                 cuda=True, return_missingness=False, test=False):
        super().__init__()
        if A is not None:
            self.i0 = i0.cuda() if cuda else i0
            self.i1 = i1.cuda() if cuda else i1
            self.A = A.cuda() if cuda else A
            self.N = torch.cat([i0, i1]).unique().shape[0]
        else:
            self.i0 = None
            self.i1 = None
            self.A = None
            if X_cts is not None:
                self.N = X_cts.shape[0]
            elif X_bin is not None:
                self.N = X_bin.shape[0]
            else:
                raise ValueError("need at least some attributes if no network")
        self.X_cts = X_cts
        self.X_bin = X_bin
        self.return_missingness = return_missingness
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
        if self.X_cts is not None and cuda:
            self.X_cts.cuda()
        if self.X_bin is not None and cuda:
            self.X_bin.cuda()

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        i0 = self.i0[i]if self.i0 is not None else None
        i1 = self.i1[i] if self.i1 is not None else None
        A = self.A[i] if self.A is not None else None
        j = torch.cat([i0, i1], 0).unique() if self.A is not None else i
        X_cts = None
        X_bin = None
        if self.X_cts is not None:
            X_cts = self.X_cts[j, :]
        if self.X_bin is not None:
            X_bin = self.X_bin[j, :]
        return i0, i1, A, j, X_cts, X_bin