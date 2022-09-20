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





# =============================================================================================
from NAIVI.vmp.factors.normal_prior import NormalPrior, MultivariateNormalPrior
from NAIVI.vmp.variables import Variable

N, K, p = 10, 5, 3

child = Variable((N, K, p))
prior = MultivariateNormalPrior(dim=p)
prior.set_children(child=child)
print(prior)
# =============================================================================================









# =============================================================================================
from NAIVI.vmp.factors.affine import Affine
from NAIVI.vmp.variables import Variable

N, K, p = 10, 5, 3

parent = Variable((N, K))
child = Variable((N, p))
self = Affine(dim_in=K, dim_out=p, parent=parent)
self.set_children(child=child)
print(self)

child.compute_posterior()
child.posterior
# =============================================================================================