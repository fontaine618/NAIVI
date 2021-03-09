import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from NAIVI.vmp.bernoulli import Bernoulli
from NAIVI.vmp.gaussian import Gaussian
from NAIVI.vmp.utils import sigmoid_integrals


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
            variance: torch.Tensor
    ):
        # TODO: variance as tensor, variance in log scale
        VMPFactor.__init__(self)
        self._deterministic = False
        self._observed = False
        self._prior = False
        self.log_var = nn.Parameter(variance)
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
        out = "GaussianFactor\n"
        out += " - parent: {}\n".format(str(self.parent))
        out += " - variance: {}\n".format(str(self.log_var.data.exp()))
        out += " - child: {}".format(str(self.child))
        return out

    @classmethod
    def prior(
            cls,
            child: Gaussian,
            mean: float,
            variance: torch.Tensor
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
            variance: torch.Tensor
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
        v = message_from_parent.variance + self.log_var.exp().unsqueeze(0)
        message_to_child = Gaussian.from_array(m, v)
        self.child.update(self.message_to_child, message_to_child)
        self.message_to_child = message_to_child

    def to_parent(self):
        message_from_child = self.child / self.message_to_child
        m = message_from_child.mean
        v = message_from_child.variance + self.log_var.exp().unsqueeze(0)
        message_to_parent = Gaussian.from_array(m, v)
        self.parent.update(self.message_to_parent, message_to_parent)
        self.message_to_parent = message_to_parent

    def to_elbo(self):
        mask = torch.logical_or(self.child.is_uniform, self.parent.is_uniform)
        cm, cv = self.child.mean_and_variance
        pm, pv = self.parent.mean_and_variance
        elbo = (pv + pm ** 2 + cv + cm ** 2 - 2. * cm * pm) / self.log_var.exp().unsqueeze(0)
        elbo += np.log(2. * np.pi) + self.log_var.unsqueeze(0)
        elbo = torch.where(mask, torch.zeros_like(elbo), elbo)
        elbo = - 0.5 * torch.sum(elbo)
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
        self._observed = False
        self.parent = parent
        self.child = child
        self.message_to_child = Bernoulli.uniform(self.shape)
        self.message_to_parent = Gaussian.uniform(self.shape)

    @classmethod
    def observed(
            cls,
            parent: Gaussian,
            child: torch.tensor
    ):
        child = Bernoulli.observed(child)
        obj = cls(parent, child)
        obj._observed = True
        return obj

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
        # proba = torch.where(self.child.proba == 1., integral, 1. - integral)
        self.message_to_child = Bernoulli(integral)

    def to_elbo(self):
        m, v = self.parent.mean_and_variance
        if m.abs().gt(100.).any():
            raise RuntimeError
        t = torch.sqrt(m ** 2 + v)
        elbo = torch.sigmoid(t).log() + (self.child.proba - 0.5) * m - 0.5 * t
        elbo = torch.where(self.child.is_uniform, torch.zeros_like(elbo), elbo)
        return torch.sum(elbo)


class Linear(VMPFactor, nn.Module):

    def __init__(
            self,
            parent: Gaussian,
            child: Gaussian
    ):
        VMPFactor.__init__(self)
        self._deterministic = True
        self.parent = parent
        self.child = child
        self.in_dim = self.parent.shape[1]
        self.out_dim = self.child.shape[1]
        dtype = self.parent.precision.dtype
        self.weight = nn.Parameter(torch.randn((self.in_dim, self.out_dim), dtype=dtype))
        self.bias = nn.Parameter(torch.randn((self.out_dim,), dtype=dtype))
        self.message_to_child = Gaussian.uniform(self.child.shape)
        self.message_to_parent_sum = Gaussian.uniform(self.parent.shape)
        self.message_to_parent = Gaussian.uniform((self.parent.shape[0], self.in_dim, self.out_dim))

    @property
    def shape(self):
        return self.weight.shape

    def size(self):
        return self.weight.size()

    def __str__(self):
        out = "Linear\n"
        out += " - parent: {}\n".format(str(self.parent))
        out += " - weight: {}\n".format(str(self.weight))
        out += " - bias: {}\n".format(str(self.bias))
        out += " - child: {}".format(str(self.child))
        return out

    def forward(self):
        self.to_child()

    def backward(self):
        self.to_parent()

    def to_parent(self):
        message_from_child = self.child / self.message_to_child
        cm, cv = message_from_child.mean_and_variance

        message_from_parent = self.parent.unsqueeze(-1) / self.message_to_parent
        pm, pv = message_from_parent.mean_and_variance

        w = self.weight.unsqueeze(0)
        b = self.bias.unsqueeze(0).unsqueeze(0)
        pm_sum = (pm * w).nansum(1, keepdim=True) - pm * w
        pv_sum = (pv * w ** 2).nansum(1, keepdim=True) - pv * w ** 2
        mean = (cm.unsqueeze(1) - b - pm_sum) / w
        var = (cv.unsqueeze(1) + pv_sum) / (w ** 2)

        message_to_parent = Gaussian.from_array(mean, var)
        message_to_parent_sum = message_to_parent.product(-1)
        self.parent.update(self.message_to_parent_sum, message_to_parent_sum)
        self.message_to_parent = message_to_parent
        self.message_to_parent_sum = message_to_parent_sum

    def to_child(self):
        message_from_parent = self.parent.unsqueeze(-1) / self.message_to_parent
        m, v = message_from_parent.mean_and_variance

        mean = (m * self.weight.unsqueeze(0)).nansum(1) + self.bias.unsqueeze(0)
        var = (v * self.weight.unsqueeze(0) ** 2).nansum(1)

        message_to_child = Gaussian.from_array(mean, var)
        self.child.update(self.message_to_child, message_to_child)
        self.message_to_child = message_to_child


class Sum(VMPFactor, nn.Module):

    def __init__(
            self,
            parents: Tuple[Gaussian],
            child: Gaussian
    ):
        VMPFactor.__init__(self)
        self._deterministic = True
        self.parents = parents
        self.child = child
        self.message_to_child = Gaussian.uniform(self.child.shape)
        self.message_to_parents = tuple(Gaussian.uniform(p.shape) for p in self.parents)

    @property
    def shape(self):
        return self.child.shape

    def size(self):
        return self.child.size()

    def __str__(self):
        out = "Sum\n"
        out += " - parents:\n"
        for p in self.parents:
            out += "   - {}\n".format(str(p))
        out += " - child: {}".format(str(self.child))
        return out

    def forward(self):
        self.to_child()

    def backward(self):
        self.to_parent()

    def to_parent(self):
        message_from_child = self.child / self.message_to_child
        cm, cv = message_from_child.mean_and_variance
        message_from_parents = tuple(
            p / mtp
            for p, mtp
            in zip(self.parents, self.message_to_parents)
        )
        pm = tuple(mfp.mean for mfp in message_from_parents)
        pv = tuple(mfp.variance for mfp in message_from_parents)
        mean = torch.cat(pm, 1).nansum(1, keepdims=True)
        var = torch.cat(pv, 1).nansum(1, keepdims=True)

        mmtp = tuple(cm - mean + m for m in pm)
        vmtp = tuple(cv + var - v for v in pv)

        mtp = tuple(Gaussian.from_array(m, v) for m, v in zip(mmtp, vmtp))
        for p, mtp_prev, mtp_new in zip(self.parents, self.message_to_parents, mtp):
            p.update(mtp_prev, mtp_new)
        self.message_to_parents = mtp

    def to_child(self):
        message_from_parents = tuple(
            p / mtp
            for p, mtp
            in zip(self.parents, self.message_to_parents)
        )
        m = tuple(mfp.mean for mfp in message_from_parents)
        v = tuple(mfp.variance for mfp in message_from_parents)

        mean = torch.cat(m, 1).nansum(1, keepdims=True)
        var = torch.cat(v, 1).nansum(1, keepdims=True)

        message_to_child = Gaussian.from_array(mean, var)
        self.child.update(self.message_to_child, message_to_child)
        self.message_to_child = message_to_child


class Select(VMPFactor, nn.Module):

    def __init__(
            self,
            parent: Gaussian,
            index: torch.Tensor,
            child: Gaussian
    ):
        VMPFactor.__init__(self)
        self._deterministic = True
        self.parent = parent
        self.index = index
        self.child = child
        self.which = torch.stack([self.index == i for i in range(self.parent.shape[0])])
        self.message_to_child = Gaussian.uniform(self.child.shape)
        self.message_to_parent = Gaussian.uniform(self.child.shape)
        self.message_to_parent_sum = Gaussian.uniform(self.parent.shape)

    @property
    def shape(self):
        return self.child.shape

    def size(self):
        return self.child.size()

    def __str__(self):
        out = "Select\n"
        out += " - parent: {}\n".format(str(self.parent))
        out += " - index: {}\n".format(str(self.index))
        out += " - child: {}".format(str(self.child))
        return out

    def forward(self):
        self.to_child()

    def backward(self):
        self.to_parent()

    def to_parent(self):
        message_from_child = self.child / self.message_to_child
        p, mtp = message_from_child.natural
        p_sum = torch.zeros(self.parent.shape, device=p.device, dtype=p.dtype)
        mtp_sum = torch.zeros(self.parent.shape, device=p.device, dtype=p.dtype)

        # for i in range(self.parent.shape[0]):
        #     p_sum[i, ] = p[self.which[i, ], ].nansum(0)
        #     mtp_sum[i, ] = mtp[self.which[i, ], ].nansum(0)

        # random updates seems to help a bit?
        n = self.parent.shape[0]
        for i in set(torch.randint(0, n, (n//2, ))):
            p_sum[i, ] = p[self.which[i, ], ].nansum(0)
            mtp_sum[i, ] = mtp[self.which[i, ], ].nansum(0)

        message_to_parent = Gaussian(p, mtp)
        message_to_parent_sum = Gaussian(p_sum, mtp_sum)
        self.parent.update(self.message_to_parent_sum, message_to_parent_sum)
        self.message_to_parent = message_to_parent
        self.message_to_parent_sum = message_to_parent_sum

    def to_child(self):
        message_from_parent = self.parent[self.index, ] / self.message_to_parent

        message_to_child = message_from_parent

        self.child.update(self.message_to_child, message_to_child)
        self.message_to_child = message_to_child


class Product(VMPFactor, nn.Module):

    def __init__(
            self,
            parents: Tuple[Gaussian],
            child: Gaussian
    ):
        VMPFactor.__init__(self)
        self._deterministic = True
        self.parents = parents
        self.child = child
        self.message_to_child = Gaussian.uniform(self.child.shape)
        self.message_to_parents = tuple(Gaussian.uniform(p.shape) for p in self.parents)

    @property
    def shape(self):
        return self.child.shape

    def size(self):
        return self.child.size()

    def __str__(self):
        out = "Product\n"
        out += " - parents:\n"
        for p in self.parents:
            out += "   - {}\n".format(str(p))
        out += " - child: {}".format(str(self.child))
        return out

    def forward(self):
        self.to_child()

    def backward(self):
        self.to_parent(0)
        self.to_parent(1)

    def to_parent(self, i):
        message_from_child = self.child / self.message_to_child
        om, ov = self.parents[1 - i].mean_and_variance
        cp, cmtp = message_from_child.natural
        p = cp * (ov + om ** 2)
        mtp = cmtp * om
        message_to_parent = Gaussian(p, mtp)
        self.parents[i].update(self.message_to_parents[i], message_to_parent)
        self.message_to_parents[i].set_to(message_to_parent)

    def to_child(self):
        message_from_parents = self.parents
        m0, v0 = message_from_parents[0].mean_and_variance
        m1, v1 = message_from_parents[1].mean_and_variance
        mean = m0 * m1
        var = m0 ** 2 * v1 + m1 ** 2 * v0 + v0 * v1
        child = Gaussian.from_array(mean, var)
        message_from_child = self.child / self.message_to_child
        message_to_child = child / message_from_child
        self.child.set_to(child)
        self.message_to_child = message_to_child


class InnerProduct(VMPFactor, nn.Module):

    def __init__(
            self,
            parents: Tuple[Gaussian],
            child: Gaussian
    ):
        VMPFactor.__init__(self)
        self._deterministic = True
        self.parents = parents
        self.child = child
        self._products = Gaussian.uniform(self.parents[0].shape)
        self._product = Product(parents, self._products)
        self._linear = Linear(self._products, child)
        self._linear.weight.data = torch.ones_like(self._linear.weight.data)
        self._linear.weight.requires_grad = False
        self._linear.bias.data = torch.zeros_like(self._linear.bias.data)
        self._linear.bias.requires_grad = False

    @property
    def shape(self):
        return self.child.shape

    def size(self):
        return self.child.size()

    def __str__(self):
        out = "InnerProduct\n"
        out += " - parents:\n"
        for p in self.parents:
            out += "   - {}\n".format(str(p))
        out += " - child: {}".format(str(self.child))
        return out

    def forward(self):
        self._product.forward()
        self._linear.forward()

    def backward(self):
        self._linear.backward()
        self._product.backward()


class Split(VMPFactor, nn.Module):

    def __init__(
            self,
            parent: Gaussian,
            children: Tuple[Gaussian]
    ):
        VMPFactor.__init__(self)
        self._deterministic = True
        self.parent = parent
        self.children = children
        self._chunks = tuple(c.shape[1] for c in children)
        self.message_to_children = tuple(Gaussian.uniform(c.shape) for c in children)
        self.message_to_parent = Gaussian.uniform(self.parent.shape)

    @property
    def shape(self):
        return self.child.shape

    def size(self):
        return self.child.size()

    def __str__(self):
        out = "Split\n"
        out += " - parent: {}\n".format(str(self.parent))
        out += " - children:\n"
        for p in self.children:
            out += "   - {}\n".format(str(p))
        return out

    def forward(self):
        self.to_child()

    def backward(self):
        self.to_parent()

    def to_parent(self):
        message_from_children = tuple(
            c / m for c, m in zip(self.children, self.message_to_children)
        )
        p = tuple(c.precision for c in message_from_children)
        mtp = tuple(c.mean_times_precision for c in message_from_children)
        p = torch.cat(p, 1)
        mtp = torch.cat(mtp, 1)
        message_to_parent = Gaussian(p, mtp)
        self.parent.update(self.message_to_parent, message_to_parent)
        self.message_to_parent = message_to_parent

    def to_child(self):
        message_from_parent = self.parent / self.message_to_parent
        p, mtp = message_from_parent.natural
        ps = p.split(self._chunks, 1)
        mtps = mtp.split(self._chunks, 1)
        message_to_children = tuple(
            Gaussian(p, mtp) for p, mtp in zip(ps, mtps)
        )
        for c, m_prev, m_new in zip(self.children, self.message_to_children, message_to_children):
            c.update(m_prev, m_new)
        self.message_to_children = message_to_children
