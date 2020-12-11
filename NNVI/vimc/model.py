import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.metrics.functional import mean_squared_error
from NNVI.utils.metrics import invariant_distance, projection_distance
from NNVI.vimc.decoder import CovariateModel, AdjacencyModel
from NNVI.vimc.encoder import Encoder


class JointModel(nn.Module):

    def __init__(self, K, N, p_cts, p_bin, position_prior=(0., 1.), heterogeneity_prior=(0., 1.)):
        super().__init__()
        self.encoder = Encoder(K, N, position_prior, heterogeneity_prior)
        self.covariate_model = CovariateModel(K, p_cts, p_bin, N)
        self.adjacency_model = AdjacencyModel(N)

    def forward(self, i0, i1, iX, n_sample=1):
        p0, h0 = self.encoder(i0, n_sample)
        p1, h1 = self.encoder(i1, n_sample)
        pX, _ = self.encoder(iX, n_sample)
        mean_cts, proba_bin = self.covariate_model(pX)
        proba_adj = self.adjacency_model(
            p0, p1,
            h0, h1
        )
        return mean_cts, proba_bin, proba_adj

    def loss(self,
             mean_cts=None, X_cts=None,
             proba_bin=None, X_bin=None,
             proba_adj=None, A=None
             ):
        loss = - self.adjacency_model.log_likelihood(proba_adj, A)
        loss += - self.covariate_model.log_likelihood(mean_cts, X_cts, proba_bin, X_bin)
        loss += self.encoder.kl_divergence()
        return loss


class VIMC:

    def __init__(self, K, N, p_cts, p_bin, position_prior=(0., 1.), heterogeneity_prior=(0., 1.)):
        self.model = JointModel(K, N, p_cts, p_bin, position_prior, heterogeneity_prior)
        self.model.cuda()

    def fit(self, train, test=None, Z_true=None,
            batch_size=100, eps=1.e-6, max_iter=100,
            lr=0.001, weight_decay=0., n_sample=1):
        # compute scaling factor
        self.compute_denum(train)
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        prev_llk = -np.inf
        l = self.verbose_init()

        for epoch in range(max_iter):
            # custom split
            batches = torch.split(torch.randperm(len(train)), batch_size)
            for batch in batches:
                self.batch_update(batch, optimizer, train, n_sample)
            llk_train, out = self.epoch_metrics(Z_true, epoch, n_sample, test, train)
            if np.abs(prev_llk - llk_train) / np.abs(llk_train) < eps:
                print("-" * l)
                return out
            else:
                prev_llk = llk_train

    def verbose_init(self):
        # verbose
        form = "{:<4} |" + " {:<10}" * 3 + "|" + " {:<10}" * 2 + "|" + " {:<10}" * 3
        names = ["iter", "loss", "mse", "auroc", "inv.", "proj.", "loss", "mse", "auroc"]
        groups = ["", "Train", "", "", "Distance", "", "Test", "", ""]
        l1 = form.format(*groups)
        l2 = form.format(*names)
        l = len(l1)
        print("-" * l)
        print(l1)
        print(l2)
        print("-" * l)
        return l

    def epoch_metrics(self, Z_true, epoch, n_sample, test, train):
        with torch.no_grad():
            # training metrics
            llk_train, mse_train, auroc_train = self.evaluate(train, n_sample)
            # testing metrics
            if test is not None:
                llk_test, mse_test, auroc_test = self.evaluate(test, n_sample)
            else:
                llk_test, mse_test, auroc_test = 0., 0., 0.
            # distance
            if Z_true is not None:
                dist_inv, dist_proj = self.latent_distance(Z_true, n_sample)
            else:
                dist_inv, dist_proj = 0., 0.
        out = [epoch, llk_train, mse_train, auroc_train, dist_inv, dist_proj, llk_test, mse_test, auroc_test]
        form = "{:<4} |" + " {:<10.4f}" * 3 + "|" + " {:<10.4f}" * 2 + "|" + " {:<10.4f}" * 3
        print(form.format(*out))
        return llk_train, out

    def batch_update(self, batch, optimizer, train, n_sample=1):
        # get batch
        i0, i1, A, j, X_cts, X_bin = train[batch]
        i0 = i0.cuda()
        i1 = i1.cuda()
        A = A.cuda()
        j = j.cuda()
        if X_cts is not None:
            X_cts = X_cts.cuda()
        if X_bin is not None:
            X_bin = X_bin.cuda()
        # reset gradient
        optimizer.zero_grad()
        # forward pass
        mean_cts, proba_bin, proba_adj = self.model.forward(i0, i1, j, n_sample)
        # objective
        loss = self.model.loss(
            mean_cts, X_cts,
            proba_bin, X_bin,
            proba_adj, A
        ) / self.denum
        # compute gradients
        loss.backward()
        # take gradient step
        optimizer.step()

    def compute_denum(self, train):
        i0, i1, A, j, X_cts, X_bin = train[:]
        self.model.adjacency_model.n_links = len(i0)
        if X_cts is not None:
            self.model.covariate_model.n_cts = (~X_cts.isnan()).sum()
        if X_bin is not None:
            self.model.covariate_model.n_bin = (~X_bin.isnan()).sum()
        self.model.adjacency_model.n_links = len(i0)
        self.denum = self.model.covariate_model.n_cts + \
                     self.model.covariate_model.n_bin + \
                     self.model.adjacency_model.n_links

    def evaluate(self, data, n_sample=1):
        with torch.no_grad():
            # get data
            i0, i1, A, j, X_cts, X_bin = data[:]
            i0 = i0.cuda()
            i1 = i1.cuda()
            A = A.cuda()
            j = j.cuda()
            if X_cts is not None:
                X_cts = X_cts.cuda()
            if X_bin is not None:
                X_bin = X_bin.cuda()
            # get fitted values
            mean_cts, proba_bin, proba_adj = self.model.forward(i0, i1, j, n_sample)
            # get metrics
            llk = self.model.loss(
                mean_cts, X_cts,
                proba_bin, X_bin,
                proba_adj, A
            ).item() / self.denum
            auc, mse = self.prediction_metrics(X_bin, X_cts, mean_cts, n_sample, proba_bin)
        return llk, mse, auc

    def prediction_metrics(self, X_bin, X_cts, mean_cts, n_sample, proba_bin):
        mse = 0.
        auc = 0.
        if X_cts is not None:
            X_cts = X_cts.unsqueeze(1)
            which_cts = ~X_cts.isnan()
            for i in range(n_sample):
                mean_cts_tmp = mean_cts[:, [i], :]
                mse += mean_squared_error(mean_cts_tmp[which_cts], X_cts[which_cts]).item()
            mse = mse / n_sample
        if X_bin is not None:
            X_bin = X_bin.unsqueeze(1)
            which_bin = ~X_bin.isnan()
            for i in range(n_sample):
                proba_bin_tmp = proba_bin[:, [i], :]
                auc += auroc(proba_bin_tmp[which_bin], X_bin[which_bin]).item()
            auc = auc / n_sample
        return auc, mse

    def latent_positions(self, n_sample=1):
        i = torch.arange(self.model.adjacency_model.N).cuda()
        Z, _ = self.model.encoder(i, n_sample)
        return Z.cpu()

    def latent_heterogeneity(self, n_sample=1):
        i = torch.arange(self.model.adjacency_model.N).cuda()
        _, a = self.model.encoder(i, n_sample)
        return a.cpu()

    def latent_distance(self, Z, n_sample):
        ZZ = self.latent_positions(n_sample)
        dist_inv = 0.
        dist_proj = 0.
        for i in range(n_sample):
            dist_inv += invariant_distance(Z, ZZ[:, i, :])
            dist_proj += projection_distance(Z, ZZ[:, i, :])
        return dist_inv / n_sample, dist_proj / n_sample

