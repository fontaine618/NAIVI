import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorEncoder(nn.Module):

    def __init__(self, dim, prior=(0., 1.)):
        super().__init__()
        self.dim = dim
        self.prior_mean = torch.tensor(prior[0])
        self.prior_log_var = torch.tensor(prior[1]).log()
        self.mean_encoder = nn.Linear(dim[0], dim[1], bias=False).float()
        self.log_var_encoder = nn.Linear(dim[0], dim[1], bias=False).float()

    def forward(self, indices):
        one_hot = F.one_hot(indices.to(torch.int64), self.dim[0]).float()
        mean = self.mean_encoder(one_hot)
        var = (self.log_var_encoder(one_hot) * 2.).exp()
        return mean, var

    def kl_divergence(self):
        mean = self.mean_encoder.weight
        log_var = self.log_var_encoder.weight
        var = log_var.exp()
        prior_var = self.prior_log_var.exp()
        kl = - (log_var - self.prior_log_var) - 1.
        kl += (mean ** 2 + var - 2. * mean * self.prior_mean + self.prior_mean ** 2) / prior_var
        return 0.5 * kl.sum()


class Encoder(nn.Module):

    def __init__(self, K, N, position_prior=(0., 1.), heterogeneity_prior=(-2., 1.)):
        super().__init__()
        self.K = K
        self.N = N
        self.latent_position_encoder = PriorEncoder((N, K), position_prior)
        self.latent_heterogeneity_encoder = PriorEncoder((N, 1), heterogeneity_prior)

    def forward(self, indices):
        return self.forward_both(indices)

    def forward_both(self, indices):
        pm, pv = self.latent_position_encoder(indices)
        hm, hv = self.latent_heterogeneity_encoder(indices)
        return pm, pv, hm, hv

    def forward_position(self, indices):
        return self.latent_position_encoder(indices)

    def kl_divergence(self):
        return self.latent_heterogeneity_encoder.kl_divergence() + \
               self.latent_position_encoder.kl_divergence()