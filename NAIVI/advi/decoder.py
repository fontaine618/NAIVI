import torch
import torch.nn as nn
from NAIVI.advi.factors import Linear, AddVariance, Logistic, InnerProduct, Sum


class CovariateModel(nn.Module):

    def __init__(self, K, p_cts=0, p_bin=0, N=1.):
        super().__init__()
        self.K = K
        self.p_cts = p_cts
        self.p_bin = p_bin
        p = p_cts + p_bin
        self.p = p
        self.mean_model = Linear(K, p)
        self.model_cts = AddVariance(p_cts)
        self.model_bin = Logistic(p_bin)
        self.n_cts = 0.
        self.n_bin = 0.

    def forward(self, mean, var):
        mean, var = self.mean_model(mean, var)
        mean_cts, mean_bin = mean.split((self.p_cts, self.p_bin), dim=-1)
        var_cts, var_bin = var.split((self.p_cts, self.p_bin), dim=-1)
        return mean_cts, var_cts, mean_bin, var_bin

    def predict(self, mean, var):
        mean_cts, var_cts, mean_bin, var_bin = self.forward(mean, var)
        mean_cts, _ = self.model_cts(mean_cts, var_cts)
        proba = self.model_bin(mean_bin, var_bin)
        return mean_cts, proba

    def elbo(self, mean, var, X_cts=None, X_bin=None):
        mean_cts, var_cts, mean_bin, var_bin = self.forward(mean, var)
        elbo = 0.
        if X_cts is not None:
            elbo += self.model_cts.elbo(mean_cts, var_cts, X_cts)
        if X_bin is not None:
            elbo += self.model_bin.elbo(mean_bin, var_bin, X_bin)
        return elbo

    @property
    def weight(self):
        return self.mean_model.weight


class AdjacencyModel(nn.Module):

    def __init__(self, N=1.):
        super().__init__()
        self.inner_product = InnerProduct()
        self.sum = Sum()
        self.logistic = Logistic()
        self.n_links = 0.

    def forward(self,
                pm0, pv0,
                pm1, pv1,
                hm0, hv0,
                hm1, hv1
                ):
        ip_mean, ip_var = self.inner_product(pm0, pv0, pm1, pv1)
        mean, var = self.sum(ip_mean, ip_var, hm0, hv0, hm1, hv1)
        return mean, var

    def predict(self,
                pm0, pv0,
                pm1, pv1,
                hm0, hv0,
                hm1, hv1
                ):
        mean, var = self.forward(pm0, pv0, pm1, pv1, hm0, hv0, hm1, hv1)
        proba = self.logistic(mean, var)
        return proba

    def elbo(self,
                pm0, pv0,
                pm1, pv1,
                hm0, hv0,
                hm1, hv1,
             A):
        mean, var = self.forward(pm0, pv0, pm1, pv1, hm0, hv0, hm1, hv1)
        return self.logistic.elbo(mean, var, A)

