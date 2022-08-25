import torch
from NAIVI.vmp.distributions import Distribution
from NAIVI.vmp.distributions.normal import Normal, MultivariateNormal
from NAIVI.vmp.distributions.point_mass import PointMass

n0 = Normal.standard_from_dimension([2, 3])
n1 = Normal(torch.ones([2, 3]) * 2, torch.zeros([2, 3]))
(n1 / n0).mean_and_variance

N, d = 3, 2
self = MultivariateNormal(
	torch.stack([torch.eye(d) for _ in range(N)], 0)*2,
	torch.ones(N, d)
)

dim = [2, 3, 4]

torch.ones(N, d) * float("Inf")

self = PointMass(torch.full(dim, float("NaN")))