import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, K=1, N=10):
        super().__init__()
        self.K = K
        self.N = N
        self.latent_position_encoder = nn.Linear(N, K, bias=False).float()
        self.latent_heterogeneity_encoder = nn.Linear(N, 1, bias=False).float()
        # initialize
        self.project()

    def forward(self, indices):
        one_hot = F.one_hot(indices.to(torch.int64), self.N).float()
        latent_position = self.latent_position_encoder(one_hot)
        latent_heterogeneity = self.latent_heterogeneity_encoder(one_hot)
        return latent_position, latent_heterogeneity

    def project(self):
        with torch.no_grad():
            pos = self.latent_position_encoder.weight
            pos = pos - torch.mean(pos, 1, keepdim=True)
            self.latent_position_encoder.weight.data = pos
