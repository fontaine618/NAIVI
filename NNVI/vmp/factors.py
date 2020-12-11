import torch
import torch.nn as nn
import numpy as np
from NNVI.vmp.bernoulli import Bernoulli
from NNVI.vmp.gaussian import Gaussian
from NNVI.vmp.utils import sigmoid_integrals


class VMPFactor(nn.Module):

    def __init__(self, **kwargs):
        nn.Module.__init__(self)
        self._deterministic = True

    def to_elbo(self, **kwargs):
        if self._deterministic:
            return 0.
        else:
            raise NotImplementedError("to_elbo not implemented for this stochastic factor")

    def forward(self):
        raise NotImplementedError("forward not implemented for this factor")

    def backward(self):
        raise NotImplementedError("forward not implemented for this factor")

    def __repr__(self):
        return str(self)

    def __str__(self):
        out = "VMPFactor()"
        return out


class GaussianFactor(VMPFactor, nn.Module):

    def __init__(
            self,
            parent: Gaussian,
            child: Gaussian,
            variance: float
    ):
        VMPFactor.__init__(self)
        self._deterministic = False
        self._observed = False
        self._prior = False
        self.variance = nn.Parameter(torch.tensor(variance))
        self.parent = parent
        self.child = child
        self.message_to_child = Gaussian.uniform(self.shape)
        self.message_to_parent = Gaussian.uniform(self.shape)

    @property
    def shape(self):
        return self.parent.shape

    def size(self):
        return self.parent.size()

    def __str__(self):
        out = "GaussianFactor(variance={})\n".format(self.variance.data)
        out += " - parent: {}\n".format(str(self.parent))
        out += " - child: {}".format(str(self.child))
        return out

    @classmethod
    def prior(
            cls,
            child: Gaussian,
            mean: float,
            variance: float
    ):
        shape = child.shape
        parent = Gaussian.point_mass(torch.full(shape, mean))
        obj = cls(parent, child, variance)
        obj._prior = True
        return obj

    @classmethod
    def observed(
            cls,
            parent: Gaussian,
            child: torch.tensor,
            variance: float
    ):
        child = Gaussian.observed(child)
        obj = cls(parent, child, variance)
        obj._observed = True
        return obj

    def forward(self):
        if not self._observed:
            self.to_child()

    def backward(self):
        if not self._prior:
            self.to_parent()

    def to_child(self):
        message_from_parent = self.parent / self.message_to_parent
        m = message_from_parent.mean
        v = message_from_parent.variance + self.variance
        message_to_child = Gaussian.from_array(m, v)
        self.child.update(self.message_to_child, message_to_child)
        self.message_to_child = message_to_child

    def to_parent(self):
        message_from_child = self.child / self.message_to_child
        m = message_from_child.mean
        v = message_from_child.variance + self.variance
        message_to_parent = Gaussian.from_array(m, v)
        self.parent.update(self.message_to_parent, message_to_parent)
        self.message_to_parent = message_to_parent

    def to_elbo(self):
        mask = torch.logical_or(self.child.is_uniform, self.parent.is_uniform)
        cm, cv = self.child.mean_and_variance
        pm, pv = self.parent.mean_and_variance
        elbo = (pv + pm**2 + cv + cm**2 - 2.*cm*pm) / self.variance
        elbo += np.log(2.*np.pi) + torch.log(self.variance)
        elbo = torch.where(mask, torch.zeros_like(elbo), elbo)
        elbo = 0.5 * torch.sum(elbo)
        elbo += self.child.entropy()
        return elbo


class Logistic(VMPFactor, nn.Module):

    def __init__(
            self,
            parent: Gaussian,
            child: Bernoulli
    ):
        VMPFactor.__init__(self)
        self._deterministic = False
        self.parent = parent
        self.child = child
        self.message_to_child = Bernoulli.uniform(self.shape)
        self.message_to_parent = Gaussian.uniform(self.shape)

    @property
    def shape(self):
        return self.parent.shape

    def size(self):
        return self.parent.size()

    def __str__(self):
        out = "Logistic\n"
        out += " - parent: {}\n".format(str(self.parent))
        out += " - child: {}".format(str(self.child))
        return out

    def forward(self):
        self.to_child()

    def backward(self):
        self.to_parent()

    def to_parent(self):
        m, v = self.parent.mean_and_variance
        integrals = sigmoid_integrals(m, v, [0, 1])
        sd = torch.sqrt(v)
        exp1 = m * integrals[0] + sd * integrals[1]
        p = (exp1 - m * integrals[0]) / v
        mtp = m * p + self.child.proba - integrals[0]
        p = torch.where(
            self.child.is_uniform,
            torch.full_like(p, Gaussian.uniform_precision),
            p
        )
        mtp = torch.where(
            self.child.is_uniform,
            torch.full_like(mtp, 0.),
            mtp
        )
        message_to_parent = Gaussian(p, mtp)
        self.parent.update(self.message_to_parent, message_to_parent)
        self.message_to_parent = message_to_parent

    def to_child(self):
        message_from_parent = self.parent / self.message_to_parent
        m, v = message_from_parent.mean_and_variance
        integral = sigmoid_integrals(m, v, [0])[0]
        proba = torch.where(self.child.proba == 1., integral, 1. - integral)
        self.message_to_child = Bernoulli(proba)

    def to_elbo(self):
        m, v = self.parent.mean_and_variance
        t = torch.sqrt(m ** 2 + v)
        elbo = torch.sigmoid(t).log() + (self.child.proba - 0.5) * m - 0.5 * t
        elbo = torch.where(self.child.is_uniform, torch.zeros_like(elbo), elbo)
        return torch.sum(elbo)
