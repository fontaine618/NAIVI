import torch
import torch.nn as nn
import numpy as np
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.metrics.functional import mean_squared_error
from NNVI.utils.metrics import invariant_distance, projection_distance
from NNVI.vimc.decoder import CovariateModel, AdjacencyModel
from NNVI.vimc.encoder import Encoder
from NNVI.naivi.naivi import NAIVI


class JointModel(nn.Module):

    def __init__(self, K, N, p_cts, p_bin, n_samples=1):
        super().__init__()
        self.n_samples = n_samples
        self.encoder = Encoder(K, N)
        self.covariate_model = CovariateModel(K, p_cts, p_bin, N)
        self.adjacency_model = AdjacencyModel(N)

    def forward(self, i0, i1, iX):
        p0, h0 = self.encoder(i0, self.n_samples)
        p1, h1 = self.encoder(i1, self.n_samples)
        pX, _ = self.encoder(iX, self.n_samples)
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

    def loss_and_fitted_values(self, i0, i1, j, X_cts=None, X_bin=None, A=None):
        mean_cts, proba_bin, proba_adj = self.forward(i0, i1, j)
        loss = self.loss(mean_cts, X_cts, proba_bin, X_bin, proba_adj, A)
        return loss, mean_cts, proba_bin, proba_adj

    def project(self):
        pass


class VIMC(NAIVI):

    def __init__(self, K, N, p_cts, p_bin, n_samples=1):
        self.model = JointModel(K, N, p_cts, p_bin, n_samples)
        self.model.cuda()

    def prediction_metrics(self, X_bin, X_cts, mean_cts, proba_bin):
        n_sample = self.model.n_samples
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
        return Z

    def latent_heterogeneity(self, n_sample=1):
        i = torch.arange(self.model.adjacency_model.N).cuda()
        _, a = self.model.encoder(i, n_sample)
        return a

    def latent_distance(self, Z):
        n_sample = self.model.n_samples
        ZZ = self.latent_positions(n_sample)
        dist_inv = 0.
        dist_proj = 0.
        for i in range(n_sample):
            dist_inv += invariant_distance(Z, ZZ[:, i, :])
            dist_proj += projection_distance(Z, ZZ[:, i, :])
        return dist_inv / n_sample, dist_proj / n_sample