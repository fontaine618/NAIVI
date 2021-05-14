import torch
import math
import torch.nn as nn


class CovariateModel(nn.Module):

    def __init__(self, K=1, p_cts=0, p_bin=0, N=1.):
        super().__init__()
        self.K = K
        self.p_cts = p_cts
        self.p_bin = p_bin
        p = p_cts + p_bin
        self.p = p
        self.mean_model = nn.Linear(K, p)
        self.cts_logvar = nn.Parameter(torch.rand((1, p_cts)))
        self.n_cts = 0.
        self.n_bin = 0.

    def forward(self, latent_positions):
        mean = self.mean_model(latent_positions)
        mean_cts, logit_bin = mean.split((self.p_cts, self.p_bin), dim=1)
        proba_bin = torch.sigmoid(logit_bin)
        return mean_cts, proba_bin

    def log_likelihood(self, mean_cts=None, X_cts=None, proba_bin=None, X_bin=None):
        nll = torch.tensor(0.).cuda()
        if (X_cts is not None) and (mean_cts is not None):
            var = torch.exp(self.cts_logvar)
            X_cts_masked = X_cts.masked_fill(X_cts.isnan(), 0.)
            llk = (mean_cts - X_cts_masked) ** 2 / (2. * var) + 0.5 * torch.log(2. * math.pi * var)
            llk = llk.masked_fill(X_cts.isnan(), 0.)
            nll += torch.nansum(llk) # * self.n_cts / max(1., (~X_cts.isnan()).sum())
        if (X_bin is not None) and (proba_bin is not None):
            X_bin_masked = X_bin.masked_fill(X_bin.isnan(), 0.)
            llk = X_bin_masked * torch.log(proba_bin) + (1.-X_bin_masked) * torch.log(1. - proba_bin)
            llk = llk.masked_fill(X_bin.isnan(), 0.)
            nll += - torch.nansum(llk) # * self.n_bin / max(1., (~X_bin.isnan()).sum())
        return - nll

    @property
    def weight(self):
        return self.mean_model.weight

    @property
    def bias(self):
        return self.mean_model.bias

    # For GLM compatibility
    def loss(self,
             mean_cts=None, X_cts=None,
             proba_bin=None, X_bin=None,
             ):
        loss = - self.log_likelihood(mean_cts, X_cts, proba_bin, X_bin)
        return loss

    def loss_and_fitted_values(self, latent_positions, X_cts=None, X_bin=None):
        mean_cts, proba_bin = self.forward(latent_positions)
        loss = self.loss(mean_cts, X_cts, proba_bin, X_bin)
        return loss, mean_cts, proba_bin


class AdjacencyModel(nn.Module):

    def __init__(self, N):
        super().__init__()
        self.N = N
        self.n_links = 0.

    def forward(self,
                latent_position0, latent_position1,
                latent_heterogeneity0, latent_heterogeneity1
    ):
        inner_products = torch.sum(latent_position0 * latent_position1, 1, keepdim=True)
        logit = inner_products + latent_heterogeneity0 + latent_heterogeneity1
        proba = torch.sigmoid(logit)
        return proba

    def log_likelihood(self, proba=None, A=None):
        llk = torch.tensor(0.).cuda()
        if (A is not None) and (proba is not None):
            llk = A * torch.log(proba) + (1.-A) * torch.log(1. - proba)
        return torch.nansum(llk)