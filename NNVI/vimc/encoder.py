import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Select(nn.Module):

    def __init__(self, dim):
        super(Select, self).__init__()
        bound = 1. / math.sqrt(dim[0]*dim[1])
        self.values = nn.Parameter(torch.randn(dim) * bound, requires_grad=True)

    def forward(self, indices):
        return self.values[indices, :]


class PriorEncoder(nn.Module):

    def __init__(self, dim, prior=(0., 1.)):
        super().__init__()
        self.dim = dim
        self.prior_mean = torch.tensor(prior[0]).cuda()
        self.prior_log_sd = torch.tensor(prior[1]).log().cuda()
        self.mean_encoder = Select(dim)
        self.log_sd_encoder = Select(dim)
        with torch.no_grad():
            self.log_sd_encoder.values.data.fill_(-2.)

    def forward(self, indices, n_sample=1):
        # get mean and sd
        mean = self.mean_encoder(indices).unsqueeze(1)
        sd = self.log_sd_encoder(indices).exp().unsqueeze(1)
        # sample
        d = list(mean.size())
        d[1] = n_sample
        z = torch.randn(d).cuda()
        x = mean + z * sd
        return x

    def kl_divergence(self):
        mean = self.mean_encoder.values
        log_sd = self.log_sd_encoder.values
        var = (2. * log_sd).exp()
        prior_var = (2. * self.prior_log_sd).exp()
        kl = - 2. * (log_sd - self.prior_log_sd) - 1.
        kl += (mean ** 2 + var - 2. * mean * self.prior_mean + self.prior_mean ** 2) / prior_var
        return 0.5 * kl.sum()

    def init(self, mean):
        self.mean_encoder.values.data = mean

    @property
    def mean(self):
        return self.mean_encoder.values


class Encoder(nn.Module):

    def __init__(self, K, N, position_prior=(0., 1.), heterogeneity_prior=(0., 1.)):
        super().__init__()
        self.K = K
        self.N = N
        self.latent_position_encoder = PriorEncoder((N, K), position_prior)
        self.latent_heterogeneity_encoder = PriorEncoder((N, 1), heterogeneity_prior)

    def forward(self, indices, n_sample=1):
        latent_position = self.latent_position_encoder(indices, n_sample)
        latent_heterogeneity = self.latent_heterogeneity_encoder(indices, n_sample)
        return latent_position, latent_heterogeneity

    def kl_divergence(self):
        return self.latent_heterogeneity_encoder.kl_divergence() + \
               self.latent_position_encoder.kl_divergence()
