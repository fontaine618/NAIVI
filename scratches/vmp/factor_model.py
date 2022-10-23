import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)

import numpy as np
from NAIVI.vmp.factors.affine import Affine
from NAIVI.vmp.factors.gaussian import GaussianFactor
from NAIVI.vmp.factors.normal_prior import MultivariateNormalPrior
from NAIVI.vmp.factors.observedfactor import ObservedFactor
from NAIVI.vmp.variables.variable import Variable
from NAIVI.vmp.factors.logistic import Logistic
from NAIVI.vmp.factors.select import Select
from NAIVI.vmp.factors.sum import Sum

# for debugging
from NAIVI.vmp.factors.factor import Factor
from NAIVI.vmp.variables.variable import Variable
from NAIVI.vmp.messages.message import Message

from NAIVI.vmp.distributions.normal import MultivariateNormal, Normal

# dimensions
N = 10
K = 3
p = 100
binary = True
missing_prop = 0.

# generating values
torch.manual_seed(0)
Z = torch.randn((N, K))
b = torch.randn((p, ))
w = torch.randn((K, p))
mu = b.reshape((1, -1)) + Z @ w
X = mu + torch.randn((N, p))
if binary:
	X = torch.where(torch.sigmoid(mu) > 0.5, 1., 0.)
mask = torch.rand((N, p)) > missing_prop
Xobs = torch.where(mask, X, float('nan') * torch.ones_like(X))
Xmis = torch.where(~mask, float('nan') * torch.ones_like(X), X)

# setup factors and variables
prior = MultivariateNormalPrior(dim=K, variance=1.)
latent = Variable((N, K))
affine = Affine(K, p, latent)
mean = Variable((N, p))
if binary:
	observation_model = Logistic(p, parent=mean, method="quadratic")
else:
	observation_model = GaussianFactor(p, parent=mean)
observations = Variable((N, p))
observed = ObservedFactor(Xobs, parent=observations)

# attach children
prior.set_children(child=latent)
affine.set_children(child=mean)
observation_model.set_children(child=observations)

# initialize messages and posteriors

latent.compute_posterior()
mean.compute_posterior()
observations.compute_posterior()

prior.update_messages_from_children()
prior.update_messages_to_children()
observation_model.update_messages_from_children()
observation_model.update_messages_to_children()
observed.update_messages_from_parents()
observed.update_messages_to_parents()

# put correct parameter values
affine.parameters["bias"].data = b
affine.parameters["weights"].data = w
observation_model.parameters["variance"].data = torch.ones((p,))

# print factors to check everything ok
print(prior)
print(affine)
print(observation_model)
print(observed)

observation_model.messages_to_parents[1].damping = 1.

# update messages
with torch.no_grad():
	print((latent.posterior.mean - Z).pow(2).mean().sqrt().item())
	for _ in range(20):
		affine.update_messages_from_parents()
		affine.update_messages_to_children()
		observation_model.update_messages_from_parents()
		observation_model.update_messages_to_children()
		observation_model.update_messages_from_children()
		observation_model.update_messages_to_parents()
		affine.update_messages_from_children()
		affine.update_messages_to_parents()

		print((latent.posterior.mean - Z).pow(2).mean().sqrt().item())



# playing with the select factor
indices = torch.randint(0, N, (7, ))
select = Select(indices, parent=latent, dist=MultivariateNormal)
select.update_messages_from_parents()

select.messages_to_parents[0].message_to_factor.mean
latent.posterior.mean
indices

select = Select(indices, parent=mean, dist=Normal)
select.update_messages_from_parents()
select.messages_to_parents[1].message_to_factor.mean[:, 0]
mean.posterior.mean[:, 0]
indices


# play with Sum factor
import torch
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from NAIVI.vmp.variables.variable import Variable
from NAIVI.vmp.factors.sum import Sum
from NAIVI.vmp.distributions.normal import Normal

dim = (2, 3)
p1 = Variable(dim)
p2 = Variable(dim)
c = Variable(dim)

self = Sum(p1=p1, p2=p2)
self.set_children(child=c)

self.messages_to_parents[0].message_to_factor = Normal.from_mean_and_variance(
	mean=torch.randn(dim), variance=torch.randn(dim)**2
)
self.messages_to_parents[1].message_to_factor = Normal.from_mean_and_variance(
	mean=torch.randn(dim), variance=torch.randn(dim)**2
)
self.messages_to_children[2].message_to_factor = Normal.from_mean_and_variance(
	mean=torch.randn(dim), variance=torch.randn(dim)**2
)

self.update_messages_to_children()
self.update_messages_to_parents()
c.posterior.mean_and_variance
p1.posterior.mean_and_variance
p2.posterior.mean_and_variance





#
from sklearn.linear_model import LogisticRegression
i = 1
logreg = LogisticRegression(penalty="l2", C=1.)
logreg.fit(w.cpu().numpy().T, Xobs.cpu().numpy()[i, :])
logreg.coef_
latent.posterior.mean[i, :]
Z[i, :]


import torch
indices = torch.Tensor([0, 1, 0, 3, 2, 2])
mean = torch.randn((4, 3))

out = mean.index_select(0, indices.long())

m0 = torch.zeros_like(mean)
m0.index_add_(0, indices.long(), out)

# TODO: think about damping?

# Print the current state of the model
def print_message(msg, vtf=True):
	print(" "*39 + "A |")
	print(" "*39 + "| |")
	print(" "*39 + "| |")
	print(
		f"{str(msg.message_to_variable if vtf else msg.message_to_factor):>38} | |"
	)
	print(" "*39 + "| |")
	print(" "*39 + "| |")
	print(str(msg).center(81))
	print(" "*39 + "| |")
	print(" "*39 + "| |")
	print(
		f"{'':38} | | {str(msg.message_to_factor if vtf else msg.message_to_variable):<38}"
	)
	print(" "*39 + "| |")
	print(" "*39 + "| |")
	print(" "*39 + "| V")


def print_variable(var):
	print(
		f"{str(var):>38}  \u23FA  {str(var.posterior):<38}"
	)
def print_factor(factor):
	print(
		f"{repr(factor):>38}  \u23F9"
	)



print_factor(prior)
print_message(prior.messages_to_children[latent.id], vtf=False)
print_variable(latent)
print_message(affine.messages_to_parents[latent.id], vtf=True)
print_factor(affine)
print_message(affine.messages_to_children[mean.id], vtf=False)
print_variable(mean)
print_message(observation_model.messages_to_parents[mean.id], vtf=True)
print_factor(observation_model)
print_message(observation_model.messages_to_children[observations.id], vtf=False)
print_variable(observations)
print_message(observed.messages_to_parents[observations.id], vtf=True)
print_factor(observed)

# inspect messages

self = Message.instance[5]
self.message_to_factor.proba
self.message_to_variable.proba

self = Message.instance[2]
self.message_to_variable.value

self = Message.instance[1]
self.message_to_variable.mean
self.message_to_factor.precision

self = observation_model


self = Message.instance[4]
self.message_to_variable.mean_times_precision
self.message_to_factor.mean_times_precision

self=affine


self = Message.instance[0]
self.message_to_variable.shape
self.message_to_factor.shape



# Tilted

mean = torch.randn((3, 4))
variance = torch.randn((3, 4)).pow(2)

_tilted_fixed_point(mean, variance)



# checking integration

mean = torch.randn((1, ))
variance = torch.randn((1, )).pow(2)
sd = variance.sqrt()


z = torch.randn((10000, 1))
_ms_expit_moment(0, mean, variance)
torch.sigmoid(z*sd + mean).mean()
_ms_expit_moment(1, mean, variance)
(torch.sigmoid(z*sd + mean)*z).mean()