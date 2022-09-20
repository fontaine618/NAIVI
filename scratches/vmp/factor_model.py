import torch
import numpy as np
from NAIVI.vmp.factors.affine import Affine
from NAIVI.vmp.factors.gaussian import GaussianFactor
from NAIVI.vmp.factors.normal_prior import MultivariateNormalPrior
from NAIVI.vmp.factors.observedfactor import ObservedFactor
from NAIVI.vmp.variables.variable import Variable

torch.set_default_tensor_type(torch.cuda.FloatTensor)

# for debugging
from NAIVI.vmp.factors.factor import Factor
from NAIVI.vmp.variables.variable import Variable
from NAIVI.vmp.messages.message import Message

from NAIVI.vmp.distributions.normal import MultivariateNormal

# dimensions
N = 100
K = 3
p = 100

# generating values
torch.manual_seed(0)
Z = torch.randn((N, K))
b = torch.randn((p, ))
w = torch.randn((K, p))
mu = b.reshape((1, -1)) + Z @ w
X = mu + torch.randn((N, p))
mask = torch.randint(0, 2, (N, p)) == 1
Xobs = torch.where(mask, X, float('nan') * torch.ones_like(X))
Xmis = torch.where(~mask, float('nan') * torch.ones_like(X), X)

# setup factors and variables
prior = MultivariateNormalPrior(dim=K)
latent = Variable((N, K))
affine = Affine(K, p, latent)
mean = Variable((N, p))
gaussian = GaussianFactor(p, parent=mean)
observations = Variable((N, p))
observed = ObservedFactor(Xobs, parent=observations)

# attach children
prior.set_children(child=latent)
affine.set_children(child=mean)
gaussian.set_children(child=observations)

# initialize messages and posteriors

latent.compute_posterior()
mean.compute_posterior()
observations.compute_posterior()

prior.update_messages_from_children()
prior.update_messages_to_children()
gaussian.update_messages_from_children()
gaussian.update_messages_to_children()
observed.update_messages_from_parents()
observed.update_messages_to_parents()

# put correct parameter values
affine.parameters["bias"].data = b
affine.parameters["weights"].data = w
gaussian.parameters["variance"].data = torch.ones((p, ))

# print factors to check everything ok
print(prior)
print(affine)
print(gaussian)
print(observed)

# update messages
with torch.no_grad():
	for _ in range(10):
		observations.compute_posterior()
		gaussian.update_messages_from_children()
		gaussian.update_messages_to_parents()
		mean.compute_posterior()
		affine.update_messages_from_children()
		affine.update_messages_to_parents()
		latent.compute_posterior()
		affine.update_messages_from_parents()
		affine.update_messages_to_children()
		mean.compute_posterior()
		gaussian.update_messages_from_parents()
		gaussian.update_messages_to_children()
		observations.compute_posterior()

		print((latent.posterior.mean - Z).pow(2).mean().sqrt())



# Logistic
from NAIVI.vmp.factors.logistic import _ms_expit_moment
import torch

mean = torch.zeros((3, 4)) + 1.
variance = torch.ones((3, 4)) * 2.

_ms_expit_moment(0, mean, variance)
