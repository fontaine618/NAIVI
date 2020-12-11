import torch
from NNVI.vmp.gaussian import Gaussian

p = torch.ones((2, 3)) * 2.
mtp = torch.ones((2, 3)) * 3.


self = Gaussian(p, mtp)
other = Gaussian(p*0.2, mtp*0.3)
other2 = Gaussian(p*0.5, mtp*0.5)

self.update(other, other2)

self[1].update(other[1], other2[1])

self *= other

p = torch.ones((3, )) * 4.
mtp = torch.ones((3, )) * 5.

self[0, :] = Gaussian(p, mtp)
self.entropy()
self.negative_entropy()
Gaussian.point_mass(p)

self.cuda()

self.split((1, 2), 1)

Gaussian.cat([self, other, other2], 0)


import torch
from NNVI.vmp.bernoulli import Bernoulli

self = Bernoulli(torch.rand((3, 2))).cuda()
self.entropy()
self.is_point_mass
self.is_uniform
self[1, 1] = Bernoulli(torch.tensor(0.5))
self[0, 1] = Bernoulli(torch.tensor(1.0))
self[1, 0] = Bernoulli(torch.tensor(0.0))
self
self.entropy()