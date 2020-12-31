import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.metrics.functional import mean_squared_error
from NNVI.utils.metrics import invariant_distance, projection_distance
from NNVI.advi.decoder import CovariateModel, AdjacencyModel
from NNVI.advi.encoder import Encoder


class JointModel(nn.Module):

    def __init__(self, K, N, p_cts, p_bin):
        super().__init__()
        self.encoder = Encoder(K, N)
        self.covariate_model = CovariateModel(K, p_cts, p_bin, N)
        self.adjacency_model = AdjacencyModel(N)

    def forward(self, i0, i1, iX):
        pm0, pv0, hm0, hv0, pm1, pv1, hm1, hv1, pmx, pvx = self.encode(i0, i1, iX)
        mean_cts, var_cts, mean_bin, var_bin = self.covariate_model(pmx, pvx)
        mean_adj, var_adj = self.adjacency_model(pm0, pv0, pm1, pv1, hm0, hv0, hm1, hv1)
        return mean_cts, var_cts, mean_bin, var_bin, mean_adj, var_adj

    def encode(self, i0, i1, iX):
        pm0, pv0, hm0, hv0 = self.encoder(i0)
        pm1, pv1, hm1, hv1 = self.encoder(i1)
        pmx, pvx = self.encoder.forward_position(iX)
        return pm0, pv0, hm0, hv0, pm1, pv1, hm1, hv1, pmx, pvx

    def predict(self, i0, i1, iX):
        pm0, pv0, hm0, hv0, pm1, pv1, hm1, hv1, pmx, pvx = self.encode(i0, i1, iX)
        mean_cts, proba_bin = self.covariate_model.predict(pmx, pvx)
        proba_adj = self.adjacency_model.predict(pm0, pv0, pm1, pv1, hm0, hv0, hm1, hv1)
        return mean_cts, proba_bin, proba_adj

    def elbo(self, i0, i1, iX, X_cts=None, X_bin=None, A=None):
        pm0, pv0, hm0, hv0, pm1, pv1, hm1, hv1, pmx, pvx = self.encode(i0, i1, iX)
        elbo = 0.
        elbo += self.covariate_model.elbo(pmx, pvx, X_cts, X_bin)
        elbo += self.adjacency_model.elbo(pm0, pv0, pm1, pv1, hm0, hv0, hm1, hv1, A)
        elbo -= self.encoder.kl_divergence()
        return - elbo


class ADVI:

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
            n_batches = len(batches)
            for batch in batches:
                # get batch
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

    def init(self, positions=None, heterogeneity=None):
        if positions is not None:
            self.model.encoder.latent_position_encoder.mean_encoder.values.data = positions
        if heterogeneity is not None:
            self.model.encoder.latent_heterogeneity_encoder.mean_encoder.values.data = heterogeneity

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

    def batch_update(self, batch, optimizer, train):
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
        # objective
        loss = self.model.elbo(i0, i1, j, X_cts, X_bin, A) / self.denum
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
            mean_cts, proba_bin, proba_adj = self.model.predict(i0, i1, j)
            # get metrics
            llk = self.model.elbo(i0, i1, j, X_cts, X_bin, A).item() / self.denum
            auc, mse = self.prediction_metrics(X_bin, X_cts, mean_cts, proba_bin)
        return llk, mse, auc

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
        return self.model.encoder.latent_position_encoder.mean_encoder.values.cpu()

    def latent_heterogeneity(self):
        return self.model.encoder.latent_heterogeneity_encoder.mean_encoder.values.cpu()

    def latent_distance(self, Z):
        ZZ = self.latent_positions()
        return invariant_distance(Z, ZZ), projection_distance(Z, ZZ)

