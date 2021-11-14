import torch.nn as nn
from NAIVI.mle.decoder import CovariateModel, AdjacencyModel
from NAIVI.mle.encoder import Encoder
from NAIVI.naivi.naivi import NAIVI


class JointModel(nn.Module):

    def __init__(self, K, N, p_cts, p_bin, mnar=False,
				 position_prior=(0., 1.),
				 heterogeneity_prior=(-2., 1.)
                 ):
        super().__init__()
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

    def loss_and_fitted_values(self, i0, i1, j, X_cts=None, X_bin=None, A=None):
        mean_cts, proba_bin, proba_adj = self.forward(i0, i1, j)
        loss = self.loss(mean_cts, X_cts, proba_bin, X_bin, proba_adj, A)
        return loss, mean_cts, proba_bin, proba_adj

    def project(self):
        self.encoder.project()


class MLE(NAIVI):

    def __init__(self, K, N, p_cts, p_bin, mnar=False,
				 position_prior=(0., 1.),
				 heterogeneity_prior=(-2., 1.)
                 ):
        super().__init__()
        self.model = JointModel(K, N, p_cts, p_bin, mnar, position_prior, heterogeneity_prior)
        self.model.cuda()


class JointModelMAP(JointModel):

    def project(self):
        pass  # no need to project if we regularize

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


class MAP(NAIVI):

    def __init__(self, K, N, p_cts, p_bin, mnar=False,
				 position_prior=(0., 1.),
				 heterogeneity_prior=(-2., 1.)
                 ):
        super().__init__()
        self.model = JointModelMAP(K, N, p_cts, p_bin, mnar, position_prior, heterogeneity_prior)
        self.model.cuda()

