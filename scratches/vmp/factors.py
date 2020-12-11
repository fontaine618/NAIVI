# -----------------------------------------------------------------------------
# GaussianFactor
import torch
import numpy as np
from NNVI.vmp.gaussian import Gaussian
from NNVI.vmp.factors import GaussianFactor

# stochastic case
shape = (2, 3)
parent = Gaussian.from_shape(shape, 0., 1.)
child = Gaussian.from_shape(shape, 1., 2.)
self = GaussianFactor(parent, child, 1.)
self.forward()
self.backward()
self
self.to_elbo()

# prior case
shape = (2, 3)
child = Gaussian.from_shape(shape, 1., 2.)
self = GaussianFactor.prior(child, 0., 1.)
self.forward()
self.backward()
self
self.to_elbo()

# observed case

shape = (2, 3)
parent = Gaussian.from_shape(shape, 0., 1.)
child = torch.randn(shape)
child[0, 0] = np.nan
Gaussian.observed(child)
self = GaussianFactor.observed(parent, child, 1.)
self.forward()
self.backward()
self
self.to_elbo()



# -----------------------------------------------------------------------------
# Logistic
import torch
import numpy as np
from NNVI.vmp.utils import sigmoid_integrals
from NNVI.vmp.bernoulli import Bernoulli
from NNVI.vmp.gaussian import Gaussian
from NNVI.vmp.factors import Logistic

shape = (3, 2)
mean = torch.randn(shape)
variance = torch.rand(shape)
# sigmoid_integrals(mean, variance, [0,1,2])

parent = Gaussian.from_array(mean, variance)
child = (torch.rand(shape) > 0.5).float()
child[0, 0] = np.nan
child = Bernoulli.observed(child)

self = Logistic(parent, child)

self
self.forward()
self.backward()
self

self.to_elbo()