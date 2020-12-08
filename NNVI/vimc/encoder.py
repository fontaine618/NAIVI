import torch
import torch.nn as nn
import torch.nn.functional as F


class PriorEncoder(nn.Module):

    def __init__(self, dim, prior=(0., 1.)):
        super().__init__()
        self.dim = dim
        self.prior_mean = torch.tensor(prior[0]).cuda()
        self.prior_log_sd = torch.tensor(prior[1]).log().cuda()
        self.mean_encoder = nn.Linear(dim[0], dim[1], bias=False).float()
        self.log_sd_encoder = nn.Linear(dim[0], dim[1], bias=False).float()

    def forward(self, one_hot, n_sample=1):
        # get mean and sd
        mean = self.mean_encoder(one_hot).unsqueeze(1)
        sd = self.log_sd_encoder(one_hot).exp().unsqueeze(1)
        # sample
        d = list(mean.size())
        d[1] = n_sample
        z = torch.randn(d).cuda()
        x = mean + z * sd
        return x

    def kl_divergence(self):
        mean = self.mean_encoder.weight
        log_sd = self.log_sd_encoder.weight
        var = (2. * log_sd).exp()
        prior_var = (2. * self.prior_log_sd).exp()
        kl = - 2. * (log_sd - self.prior_log_sd) - 1.
        kl += (mean ** 2 + var - 2. * mean * self.prior_mean + self.prior_mean ** 2) / prior_var
        return 0.5 * kl.sum()


class Encoder(nn.Module):

    def __init__(self, K, N, position_prior=(0., 1.), heterogeneity_prior=(0., 1.)):
        super().__init__()
        self.K = K
        self.N = N
        self.latent_position_encoder = PriorEncoder((N, K), position_prior)
        self.latent_heterogeneity_encoder = PriorEncoder((N, 1), heterogeneity_prior)

    def forward(self, indices, n_sample=1):
        one_hot = F.one_hot(indices.to(torch.int64), self.N).float()
        latent_position = self.latent_position_encoder(one_hot, n_sample)
        latent_heterogeneity = self.latent_heterogeneity_encoder(one_hot, n_sample)
        return latent_position, latent_heterogeneity

    def kl_divergence(self):
        return self.latent_heterogeneity_encoder.kl_divergence() + \
               self.latent_position_encoder.kl_divergence()
