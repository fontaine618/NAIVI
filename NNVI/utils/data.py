import torch
from torch.utils.data.dataloader import Dataset


class JointDataset(Dataset):

    def __init__(self, i0, i1, A, X_cts=None, X_bin=None, cuda=True):
        super().__init__()
        if cuda:
            self.i0 = i0.cuda()
            self.i1 = i1.cuda()
            self.A = A.cuda()
            self.X_cts = X_cts
            self.X_bin = X_bin
            self.N = len(i0)
            if X_cts is not None:
                self.p_cts = X_cts.size(1)
                self.X_cts.cuda()
            if X_bin is not None:
                self.p_bin = X_bin.size(1)
                self.X_bin.cuda()
        else:
            self.i0 = i0.cpu()
            self.i1 = i1.cpu()
            self.A = A.cpu()
            self.X_cts = X_cts
            self.X_bin = X_bin
            self.N = len(i0)
            if X_cts is not None:
                self.p_cts = X_cts.size(1)
                self.X_cts.cpu()
            if X_bin is not None:
                self.p_bin = X_bin.size(1)
                self.X_bin.cpu()

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