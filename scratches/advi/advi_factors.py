import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from NNVI.advi.encoder import PriorEncoder, Encoder
from NNVI.advi.factors import Linear, AddVariance, Logistic, InnerProduct, Sum

N = 7
K = 3
p = 5




encoder = PriorEncoder((N, K)).cuda()
indices = torch.tensor([0, 2, 4, 6]).cuda()
mean, var = encoder(indices)

self = Linear(K, p).cuda()

out_mean, out_var = self(mean, var)


# cts
self = AddVariance(p).cuda()

mean, var = self(out_mean, out_var)

y = torch.randn_like(mean)
y[1, 2] = np.nan
loss = self.elbo(mean, var, y)

loss.backward()

self.log_var.grad

# bin
self = Logistic().cuda()

proba = self(mean, var)

y = torch.where(torch.rand_like(mean) < proba, 1., 0.)
y[0, 0] = np.nan

self.elbo(mean, var, y)


# IP
m0 = torch.randn((N, K))
v0 = torch.rand((N, K))
m1 = torch.randn((N, K))
v1 = torch.rand((N, K))

self = InnerProduct()
self(m0, v0, m1, v1)

# Sum
m0 = torch.randn((N, ))
v0 = torch.rand((N, ))
m1 = torch.randn((N, ))
v1 = torch.rand((N, ))
m2 = torch.randn((N, ))
v2 = torch.rand((N, ))


self = Sum()
self(m0, v0, m1, v1, m2, v2)



