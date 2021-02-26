import torch.nn as nn
from NNVI.advi.decoder import CovariateModel, AdjacencyModel
from NNVI.advi.encoder import Encoder
from NNVI.naivi.naivi import NAIVI


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

    def loss_and_fitted_values(self, i0, i1, j, X_cts=None, X_bin=None, A=None):
        mean_cts, proba_bin, proba_adj = self.predict(i0, i1, j)
        loss = self.elbo(i0, i1, j, X_cts, X_bin, A)
        return loss, mean_cts, proba_bin, proba_adj

    def project(self):
        pass


class ADVI(NAIVI):

    def __init__(self, K, N, p_cts, p_bin):
        self.model = JointModel(K, N, p_cts, p_bin)
        self.model.cuda()

