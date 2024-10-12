import torch.nn as nn
import torch
from NAIVI.mle.decoder import CovariateModel, AdjacencyModel
from NAIVI.mle.encoder import Encoder
from NAIVI.gradient_based.gradient_based import GradientBased


class JointModel(nn.Module):

    def __init__(self, K, N, p_cts, p_bin, mnar=False, network_weight=1.0,
				 position_prior=(0., 1.),
				 heterogeneity_prior=(-2., 1.),
                estimate_components=False
                 ):
        super().__init__()
        self.K = K
        self.network_weight = network_weight
        self.heterogeneity_prior = heterogeneity_prior
        self.position_prior = position_prior
        self.mnar = mnar
        self.p_cts = p_cts
        self.p_bin_og = p_bin
        if mnar:
            p_bin += p_bin + p_cts
        self.p_bin = p_bin
        self.encoder = Encoder(K, N)
        self.covariate_model = CovariateModel(K, p_cts, p_bin, N)
        self.adjacency_model = AdjacencyModel(K, estimate_components)

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
        loss = - self.adjacency_model.log_likelihood(proba_adj, A) * self.network_weight
        loss += - self.covariate_model.log_likelihood(mean_cts, X_cts, proba_bin, X_bin)
        return loss

    def loss_and_fitted_values(self, i0, i1, j, X_cts=None, X_bin=None, A=None):
        mean_cts, proba_bin, proba_adj = self.forward(i0, i1, j)
        loss = self.loss(mean_cts, X_cts, proba_bin, X_bin, proba_adj, A)
        return loss, mean_cts, proba_bin, proba_adj

    def project(self):
        pass

class JointModelMLE(JointModel):

    def loss(self,
             mean_cts=None, X_cts=None,
             proba_bin=None, X_bin=None,
             proba_adj=None, A=None
             ):
        loss = super(JointModelMLE, self).loss(
             mean_cts=mean_cts, X_cts=X_cts,
             proba_bin=proba_bin, X_bin=X_bin,
             proba_adj=proba_adj, A=A
        )
        # convex relaxation to the constraint JZ=Z
        Z = self.encoder.latent_position_encoder.mean
        Zcentered = Z - Z.mean(0, keepdim=True)
        loss += 1e6 *(Z-Zcentered).pow(2).mean()

        return loss

    def project(self):
        pass # no need to project with convex relaxation
        # self.encoder.project()


class JointModelMAP(JointModel):

    def loss(self,
             mean_cts=None, X_cts=None,
             proba_bin=None, X_bin=None,
             proba_adj=None, A=None
             ):
        loss = super(JointModelMAP, self).loss(
             mean_cts=mean_cts, X_cts=X_cts,
             proba_bin=proba_bin, X_bin=X_bin,
             proba_adj=proba_adj, A=A
        )
        # add penalty
        Z = self.encoder.latent_position_encoder.mean - \
            self.position_prior[0]
        loss += (Z**2).sum() / (2. * self.position_prior[1])
        alpha = self.encoder.latent_heterogeneity_encoder.mean - \
            self.heterogeneity_prior[0]
        loss += (alpha**2).sum() / (2. * self.heterogeneity_prior[1])

        return loss

    def project(self):
        pass  # no need to project if we regularize

class MLE(GradientBased):

    def __init__(self, K, N, p_cts, p_bin, mnar=False, network_weight=1.0,
				 position_prior=(0., 1.),
				 heterogeneity_prior=(-2., 1.),
                estimate_components=False, **kwargs
                 ):
        super().__init__()
        self.model = JointModelMLE(K, N, p_cts, p_bin, mnar, network_weight,
                                position_prior, heterogeneity_prior,
            estimate_components=estimate_components)
        if torch.cuda.is_available():
            self.model.cuda()


class MAP(GradientBased):

    def __init__(self, K, N, p_cts, p_bin, mnar=False, network_weight=1.0,
				 position_prior=(0., 1.),
				 heterogeneity_prior=(-2., 1.),
                    estimate_components=False, **kwargs
                 ):
        super().__init__()
        self.model = JointModelMAP(K, N, p_cts, p_bin, mnar, network_weight,
                                   position_prior, heterogeneity_prior,
            estimate_components=estimate_components)
        if torch.cuda.is_available():
            self.model.cuda()

