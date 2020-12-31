import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, K=1, N=10):
        super().__init__()
        self.K = K
        self.N = N
        self.latent_position_encoder = Select(N, K)
        self.latent_heterogeneity_encoder = Select(N, 1)
        # initialize
        self.project()

    def forward(self, indices):
        latent_position = self.latent_position_encoder(indices)
        latent_heterogeneity = self.latent_heterogeneity_encoder(indices)
        return latent_position, latent_heterogeneity

    def project(self):
        self.latent_position_encoder.project()


class Select(nn.Module):

    def __init__(self, N, K):
        super(Select, self).__init__()
        self.values = nn.Parameter(torch.randn(N, K) / math.sqrt(N * K), requires_grad=True)

    def forward(self, indices):
        return self.values[indices, :]

    def project(self):
        with torch.no_grad():
            pos = self.values
            pos = pos - torch.mean(pos, 1, keepdim=True)
            self.values.data = pos
