import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.metrics.functional import mean_squared_error
from NNVI.utils.metrics import invariant_distance, projection_distance
from NNVI.mle.decoder import CovariateModel, AdjacencyModel
from NNVI.mle.encoder import Encoder


class JointModel(nn.Module):

    def __init__(self, K, N, p_cts, p_bin):
        super().__init__()
        self.encoder = Encoder(K, N)
        self.covariate_model = CovariateModel(K, p_cts, p_bin, N)
        self.adjacency_model = AdjacencyModel(N)

    def forward(self, indices0, indices1, indicesX):
        latent_position0, latent_heterogeneity0 = self.encoder(indices0)
        latent_position1, latent_heterogeneity1 = self.encoder(indices1)
        latent_positionX, _ = self.encoder(indicesX)
        mean_cts, proba_bin = self.covariate_model(latent_positionX)
        proba_adj = self.adjacency_model(
            latent_position0, latent_position1,
            latent_heterogeneity0, latent_heterogeneity1
        )
        return mean_cts, proba_bin, proba_adj

    def loss(self,
             mean_cts=None, X_cts=None,
             proba_bin=None, X_bin=None,
             proba_adj=None, A=None
             ):
        loss = - self.adjacency_model.log_likelihood(proba_adj, A)
        loss += - self.covariate_model.log_likelihood(mean_cts, X_cts, proba_bin, X_bin)
        return loss

    def project(self):
        self.encoder.project()


class MLE:

    def __init__(self, K, N, p_cts, p_bin):
        self.model = JointModel(K, N, p_cts, p_bin)
        self.model.cuda()

    def fit(self, train, test=None, Z_true=None,
            batch_size=100, eps=1.e-6, max_iter=100,
            lr=0.001, weight_decay=0.):
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
                self.batch_update(batch, optimizer, train)
            llk_train, out = self.epoch_metrics(Z_true, epoch, test, train)
            if np.abs(prev_llk - llk_train) / np.abs(llk_train) < eps:
                print("-" * l)
                break
            else:
                prev_llk = llk_train
        return out

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

    def epoch_metrics(self, Z_true, epoch, test, train):
        with torch.no_grad():
            # training metrics
            llk_train, mse_train, auroc_train = self.evaluate(train)
            # testing metrics
            if test is not None:
                llk_test, mse_test, auroc_test = self.evaluate(test)
            else:
                llk_test, mse_test, auroc_test = 0., 0., 0.
            # distance
            if Z_true is not None:
                dist_inv, dist_proj = self.latent_distance(Z_true)
            else:
                dist_inv, dist_proj = 0., 0.
        out = [epoch, llk_train, mse_train, auroc_train, dist_inv, dist_proj, llk_test, mse_test, auroc_test]
        form = "{:<4} |" + " {:<10.4f}" * 3 + "|" + " {:<10.4f}" * 2 + "|" + " {:<10.4f}" * 3
        print(form.format(*out))
        return llk_train, out

    def init(self, positions=None, heterogeneity=None, bias=None, weight=None):
        if positions is not None:
            self.model.encoder.latent_position_encoder.values.data = positions
        if heterogeneity is not None:
            self.model.encoder.latent_heterogeneity_encoder.values.data = heterogeneity
        if bias is not None:
            self.model.covariate_model.mean_model.bias.data = bias.view(-1)
        if weight is not None:
            self.model.covariate_model.mean_model.weight.data = weight.t()

    def batch_update(self, batch, optimizer, train):
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
        mean_cts, proba_bin, proba_adj = self.model.forward(i0, i1, j)
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
        # center
        self.model.project()

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

    def evaluate(self, data):
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
            mean_cts, proba_bin, proba_adj = self.model.forward(i0, i1, j)
            # get metrics
            llk = self.model.loss(
                mean_cts, X_cts,
                proba_bin, X_bin,
                proba_adj, A
            ) / self.denum
            auc, mse = self.prediction_metrics(X_bin, X_cts, mean_cts, proba_bin)
        return llk.item(), mse, auc

    def prediction_metrics(self, X_bin, X_cts, mean_cts, proba_bin):
        mse = 0.
        auc = 0.
        if X_cts is not None:
            which_cts = ~X_cts.isnan()
            mse = mean_squared_error(mean_cts[which_cts], X_cts[which_cts]).item()
        if X_bin is not None:
            which_bin = ~X_bin.isnan()
            auc = auroc(proba_bin[which_bin], X_bin[which_bin]).item()
        return auc, mse

    def latent_positions(self):
        return self.model.encoder.latent_position_encoder.values.data

    def latent_heterogeneity(self):
        return self.model.encoder.latent_heterogeneity_encoder.values.data

    def latent_distance(self, Z):
        ZZ = self.latent_positions()
        return invariant_distance(Z, ZZ), projection_distance(Z, ZZ)

