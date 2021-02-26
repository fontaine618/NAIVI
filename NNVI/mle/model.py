import torch
import torch.nn as nn

from NNVI.mle.decoder import CovariateModel, AdjacencyModel
from NNVI.mle.encoder import Encoder
from NNVI.naivi.naivi import NAIVI


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

    def loss_and_fitted_values(self, i0, i1, j, X_cts=None, X_bin=None, A=None):
        mean_cts, proba_bin, proba_adj = self.forward(i0, i1, j)
        loss = self.loss(mean_cts, X_cts, proba_bin, X_bin, proba_adj, A)
        return loss, mean_cts, proba_bin, proba_adj

    def project(self):
        self.encoder.project()


class MLE(NAIVI):

    def __init__(self, K, N, p_cts, p_bin):
        super().__init__()
        self.model = JointModel(K, N, p_cts, p_bin)
        self.model.cuda()

