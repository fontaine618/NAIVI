import torch
from NAIVI.advi.encoder import PriorEncoder, Encoder


N = 10
K = 3

self = PriorEncoder((N, K)).cuda()

indices = torch.tensor([0, 2, 4, 6, 8]).cuda()

self(indices)

self.kl_divergence()


self = Encoder(K, N).cuda()

self(indices)