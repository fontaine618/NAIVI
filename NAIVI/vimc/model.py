import torch.nn as nn
from NAIVI.vimc.decoder import CovariateModel, AdjacencyModel
from NAIVI.vimc.encoder import Encoder
from NAIVI.gradient_based.gradient_based import GradientBased


class JointModel(nn.Module):

    def __init__(self, K, N, p_cts, p_bin, n_samples=1, mnar=False, network_weight=1.0,
                    position_prior=(0., 1.),
                    heterogeneity_prior=(-2., 1.),
                    estimate_components=False
    ):
        super().__init__()
        self.K = K
        self.network_weight = network_weight
        self.mnar = mnar
        self.p_cts = p_cts
        self.p_bin_og = p_bin
        if mnar:
            p_bin += p_bin + p_cts
        self.p_bin = p_bin
        self.n_samples = n_samples
        self.encoder = Encoder(K, N, position_prior, heterogeneity_prior)
        self.covariate_model = CovariateModel(K, p_cts, p_bin, N)
        self.adjacency_model = AdjacencyModel(K, estimate_components)

    def forward(self, i0, i1, iX, n_samples=None):
        if n_samples is None:
            n_samples = self.n_samples
        p0, h0 = self.encoder(i0, n_samples)
        p1, h1 = self.encoder(i1, n_samples)
        pX, _ = self.encoder(iX, n_samples)
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
        loss = - self.adjacency_model.log_likelihood(proba_adj, A) * self.network_weight
        loss += - self.covariate_model.log_likelihood(mean_cts, X_cts, proba_bin, X_bin)
        loss += self.encoder.kl_divergence()
        return loss

    def loss_and_fitted_values(self, i0, i1, j, X_cts=None, X_bin=None, A=None):
        mean_cts, proba_bin, proba_adj = self.forward(i0, i1, j)
        loss = self.loss(mean_cts, X_cts, proba_bin, X_bin, proba_adj, A)
        mean_cts, proba_bin, proba_adj = self.forward(i0, i1, j, n_samples=0)
        return loss, mean_cts.squeeze(1), proba_bin.squeeze(1), proba_adj.squeeze(1)

    def project(self):
        pass


class VIMC(GradientBased):

    def __init__(self, K, N, p_cts, p_bin, n_samples=1, mnar=False, network_weight=1.0,
                    position_prior=(0., 1.),
                    heterogeneity_prior=(-2., 1.),
                    estimate_components=False, **kwargs
    ):
        self.model = JointModel(K, N, p_cts, p_bin, n_samples, mnar, network_weight,
            position_prior=position_prior,
            heterogeneity_prior=heterogeneity_prior,
            estimate_components=estimate_components)
        self.model.cuda()