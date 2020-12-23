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
self = GaussianFactor(parent, child, torch.arange(3).double())
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



# -----------------------------------------------------------------------------
# Linear
import torch
import numpy as np
from NNVI.vmp.gaussian import Gaussian
from NNVI.vmp.factors import Linear

N = 5
K = 3
p = 4

parent = Gaussian.from_shape((N, K), 0., 1.)
child = Gaussian.from_shape((N, p), 1., 2.)

self = Linear(parent, child)

self
self.forward()
self.backward()
self




# -----------------------------------------------------------------------------
# Sum
import torch
import numpy as np
from NNVI.vmp.gaussian import Gaussian
from NNVI.vmp.factors import Sum

N = 5

p0 = Gaussian.from_shape((N, 1), 0., 1.)
p1 = Gaussian.from_shape((N, 1), 0., 1.)
p2 = Gaussian.from_shape((N, 1), 0., 1.)
parents = (p0, p1, p2)
child = Gaussian.from_shape((N, 1), 1., 2.)

self = Sum(parents, child)

self
self.forward()
self.backward()
self





# -----------------------------------------------------------------------------
# Sum
import torch
import numpy as np
from NNVI.vmp.gaussian import Gaussian
from NNVI.vmp.factors import Select

N = 5
n = N * (N-1) // 2
K = 3

parent = Gaussian.from_array(torch.randn((N, K)), torch.ones((N, K)))
index = torch.randint(0, N, (n, ))
child = Gaussian.uniform((n, K))

self = Select(parent, index, child)

self
self.forward()
self.backward()
self



# -----------------------------------------------------------------------------
# Sum
import torch
import numpy as np
from NNVI.vmp.gaussian import Gaussian
from NNVI.vmp.factors import Product

N = 5
K = 3

p0 = Gaussian.from_shape((N, K), 0., 1.)
p1 = Gaussian.from_shape((N, K), 0., 1.)
parents = (p0, p1)
child = Gaussian.from_shape((N, K), 0., 2.)

self = Product(parents, child)

self
self.forward()
self.backward()
self



# -----------------------------------------------------------------------------
# InnerProduct
import torch
import numpy as np
from NNVI.vmp.gaussian import Gaussian
from NNVI.vmp.factors import Linear
from NNVI.vmp.factors import InnerProduct

N = 5
K = 3

p0 = Gaussian.from_shape((N, K), 0., 1.)
p1 = Gaussian.from_shape((N, K), 0., 1.)
parents = (p0, p1)
child = Gaussian.from_shape((N, 1), 0., 5.)

self = InnerProduct(parents, child)

self
self.forward()
self.backward()
self



# -----------------------------------------------------------------------------
# Split
import torch
import numpy as np
from NNVI.vmp.gaussian import Gaussian
from NNVI.vmp.factors import Split

N = 5

p0 = Gaussian.from_shape((N, 2), 0., 1.)
p1 = Gaussian.from_shape((N, 4), 0., 1.)
children = (p0, p1)
parent = Gaussian.from_shape((N, 6), 0., 5.)

self = Split(parent, children)

self
self.forward()
self.backward()
self


