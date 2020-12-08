import torch
import numpy as np
from NNVI.mle.model import JointModel

N = 10
K = 2
p_cts = 3
p_bin = 5

self = JointModel(K, N, p_cts, p_bin)


indices0 = torch.tensor([[0, 1, 3, 5, 8]])
indices1 = torch.tensor([[0, 1, 3, 5, 8]])
indicesX = torch.tensor([[0, 1, 3, 5, 8]])





indices = torch.tensor([[0, 1, 3, 5, 8]])

self = Encoder(K, N)

self(indices)

self = CovariateModel(K, p_cts, p_bin)

latent_positions = torch.randn((5, K))

X_cts = torch.randn((5, p_cts))
X_cts[2, 2] = np.nan
X_bin = (torch.rand((5, p_bin)) > 0.5).float()
X_bin[2, 2] = np.nan

latent_position0 = torch.randn((5, K))
latent_position1 = torch.randn((5, K))
latent_heterogeneity0 = torch.randn((5, 1))
latent_heterogeneity1 = torch.randn((5, 1))


i = 3
i0 = torch.tensor([[0,1,2, 2]])
i1 = torch.tensor([[3, 2, 1, 1]])