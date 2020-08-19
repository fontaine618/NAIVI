import tensorflow as tf
import numpy as np
from typing import Dict
from models.distributions.gaussianarray import GaussianArray
from models.distributions.bernoulliarray import BernoulliArray
from models.parameter import ParameterArray, ParameterArrayLogScale
import tensorflow_probability as tfp


class VMPFactor:

    def __init__(self):
        self._deterministic = True
        self._parameters = dict()
        self._nodes = dict()
        self._factors = dict()

    def to_elbo(self, **kwargs):
        if self._deterministic:
            # if not defined in child, we assume deterministic node
            # and the contribution is 0
            return 0.
        else:
            # otherwise it must be implemented in the child class
            raise NotImplementedError("to_elbo not implemented for this stochastic factor")

    def parameters(self):
        parms = self._parameters
        for name, factor in self._factors.items():
            parms.update({
                name + "." + n: p
                for n, p in factor.parameters().items()
            })
        return parms

    def factors(self):
        factors = dict()
        for name, f in self._factors.items():
            factors.update({
                name + "." + n: ff
                for n, ff in f.factors().items()
            })
        factors.update(self._factors)
        return factors


class Prior(VMPFactor):
    # In the VMP, we only pass the message at the initialisation;
    # All other updates are add/remove other massages so the message from the
    # prior factor always remains.

    def __init__(self, child: GaussianArray, mean: float = 0., variance: float = 1.):
        super().__init__()
        self._deterministic = False
        self.mean = ParameterArray(mean, True, name="Prior.mean")
        self.variance = ParameterArrayLogScale(variance, True, name="Prior.variance")
        self._parameters = {"mean": self.mean, "variance": self.variance}
        self.shape = child.shape()
        self.child = child
        self.message_to_child = GaussianArray.from_shape(self.shape, self.mean.value(), self.variance.value())

    def init_child(self):
        self.child.set_to(self.message_to_child)

    def to_elbo(self):
        x = self.child
        # TODO: if we want to estimate the hyperparameters, this must me changed
        m, v = self.message_to_child.mean_and_variance()
        # model contribution
        elbo = (v + m ** 2) + -2. * x.mean() * m + (x.variance() + x.mean() ** 2)
        elbo = -.5 * tf.reduce_sum(elbo / v + tf.math.log(v))
        # node contribution
        elbo += x.entropy()
        return elbo


class AddVariance(VMPFactor):
    # Stochastic node: N(Gaussian mean, fixed variance)
    # i.e. increases the variance

    def __init__(self, child: GaussianArray, parent: GaussianArray, variance: float = 1.):
        super().__init__()
        self._deterministic = False
        self.shape = child.shape()
        self.child = child
        self.parent = parent
        self.message_to_child = GaussianArray.uniform(self.shape)
        self.message_to_parent = GaussianArray.uniform(self.shape)
        self.variance = ParameterArrayLogScale(variance, False, name="AddVariance.variance")
        self._parameters = {"variance": self.variance}

    def to_child(self):
        mean = self.parent / self.message_to_parent
        m = mean.mean()
        v = mean.variance() + self.variance.value()
        message_to_child = GaussianArray.from_array(m, v)
        self.child.update(self.message_to_child, message_to_child)
        self.message_to_child = message_to_child

    def to_parent(self):
        x = self.child / self.message_to_child
        m = x.mean()
        v = x.variance_safe() + self.variance.value()
        message_to_parent = GaussianArray.from_array(m, v)
        self.parent.update(self.message_to_parent, message_to_parent)
        self.message_to_parent = message_to_parent

    def to_elbo(self):
        mean = self.parent
        x = self.child
        xm, xv = x.mean(), x.variance_safe()
        # model contribution
        elbo = mean.variance() + mean.mean() ** 2
        elbo += xv + xm ** 2
        elbo += -2. * xm * mean.mean()
        elbo /= self.variance.value()
        # elbo += tf.math.log(2 * np.pi)
        elbo += tf.math.log(self.variance.value())
        elbo = -.5 * tf.reduce_sum(tf.where(x.is_uniform(), 0.0, elbo))
        # node contribution
        elbo += x.entropy()
        return elbo


class Probit(VMPFactor):

    def __init__(self, child: BernoulliArray, parent: GaussianArray):
        super().__init__()
        self._deterministic = True
        self.shape = child.shape()
        self.child = child
        self.parent = parent
        self.message_to_child = BernoulliArray.uniform(self.shape)
        self.message_to_parent = GaussianArray.uniform(self.shape)

    def to_parent(self):
        # TODO: assumes 0/1 only, need to fix later with missing values
        x = self.parent / self.message_to_parent
        A = self.child.proba()
        stnr = x.mean() * tf.math.sqrt(x.precision()) * tf.cast(2 * A - 1, tf.float32)
        vf = tfp.distributions.Normal(0., 1.).prob(stnr) / tfp.distributions.Normal(0., 1.).cdf(stnr)
        wf = vf * (stnr + vf)
        m = x.mean() + tf.math.sqrt(x.variance()) * vf * tf.cast(2 * A - 1, tf.float32)
        v = x.variance() * (1. - wf)
        message_to_parent = GaussianArray.from_array(m, v)
        self.parent.update(self.message_to_parent, message_to_parent)
        self.message_to_parent = message_to_parent

    def to_child(self):
        # only used for prediction, does not affect VMP
        x = self.parent / self.message_to_parent
        proba = 1. - tfp.distributions.Normal(*x.mean_and_stddev()).cdf(0.0)
        self.message_to_child = BernoulliArray.from_array(proba)


class Sum(VMPFactor):
    # sum over last dimension (could be expanded to and dim by replacing -1 <- dim with argument)

    def __init__(self, child, parent):
        super().__init__()
        self._deterministic = True
        self.child = child
        self.parent = parent
        self.message_to_child = GaussianArray.uniform(child.shape())
        self.message_to_parent = GaussianArray.uniform(parent.shape())

    def to_child(self):
        x = self.parent / self.message_to_parent
        message_to_child = GaussianArray.from_array(
            tf.math.reduce_sum(x.mean(), -1),
            tf.math.reduce_sum(x.variance(), -1)
        )
        self.child.update(self.message_to_child, message_to_child)
        self.message_to_child = message_to_child

    def to_parent(self):
        s = self.child / self.message_to_child
        x = self.parent / self.message_to_parent
        m = tf.expand_dims(s.mean(), -1) - tf.math.reduce_sum(x.mean(), -1, keepdims=True) + x.mean()
        v = tf.where(
            x.is_uniform(),
            np.inf,
            tf.expand_dims(s.variance(), -1) + tf.math.reduce_sum(x.variance(), -1, keepdims=True) - x.variance()
        )
        message_to_parent = GaussianArray.from_array(m, v)
        self.parent.update(self.message_to_parent, message_to_parent)
        self.message_to_parent = message_to_parent


class Product(VMPFactor):
    # all pairwise products in the first dimension (N, ...) => (N, N, ...)

    def __init__(self, child, parent):
        super().__init__()
        self._deterministic = True
        self.child = child
        self.parent = parent
        self.message_to_child = GaussianArray.uniform(child.shape())
        # self.message_to_parent = GaussianArray.uniform(parent.shape())
        self.message_to_parent0 = GaussianArray.uniform(parent.shape())
        self.message_to_parent1 = GaussianArray.uniform(parent.shape())

    def to_child(self):
        x0 = self.parent / self.message_to_parent0
        x1 = self.parent / self.message_to_parent1
        m0 = tf.expand_dims(x0.mean(), 0)
        m1 = tf.expand_dims(x1.mean(), 1)
        v0 = tf.expand_dims(x0.variance(), 0)
        v1 = tf.expand_dims(x1.variance(), 1)
        m = m0 * m1
        v = m0 ** 2 * v1 + m1 ** 2 * v0 + v0 * v1
        message_to_child = GaussianArray.from_array(m, v)
        self.child.update(self.message_to_child, message_to_child)
        self.message_to_child = message_to_child

    def to_parent(self):
        product = self.child / self.message_to_child

        # update 0
        x1 = self.parent / self.message_to_parent1
        p = product.precision() * tf.expand_dims(x1.variance() + x1.mean() ** 2, 1)
        mtp = product.mean_times_precision() * tf.expand_dims(x1.mean(), 1)
        p0 = tf.math.reduce_sum(p, 1)
        mtp0 = tf.math.reduce_sum(mtp, 1)

        message_to_parent0 = GaussianArray(p0, mtp0)
        self.parent.update(self.message_to_parent0, message_to_parent0)
        self.message_to_parent0 = message_to_parent0

        # update 1
        x0 = self.parent / self.message_to_parent0
        p = product.precision() * tf.expand_dims(x0.variance() + x0.mean() ** 2, 0)
        mtp = product.mean_times_precision() * tf.expand_dims(x0.mean(), 0)
        p1 = tf.math.reduce_sum(p, 0)
        mtp1 = tf.math.reduce_sum(mtp, 0)

        message_to_parent1 = GaussianArray(p1, mtp1)
        self.parent.update(self.message_to_parent1, message_to_parent1)
        self.message_to_parent1 = message_to_parent1


class WeightedSum(VMPFactor):
    # expects weighted sum over last dimension
    # x is NxK, B is KxP and B0 is 1xP
    # result is NxP

    def __init__(self, child: GaussianArray, parent: GaussianArray, bias=None, weight=None):
        super().__init__()
        self._deterministic = True
        self.child = child
        self.shape_child = child.shape()
        self.parent = parent
        self.shape_parent = parent.shape()
        self.message_to_child = GaussianArray.uniform(child.shape())
        self.message_to_parent = GaussianArray.uniform(parent.shape())
        if bias is None:
            bias = tf.zeros((1, self.shape_child[1]))
        self.bias = ParameterArray(bias, False, name="WeightedSum.bias")
        if weight is None:
            weight = tf.ones((self.shape_parent[1], self.shape_child[1]))
        self.weight = ParameterArray(weight, False, name="WeightedSum.weight")
        self._parameters = {"bias": self.bias, "weight": self.weight}

    def to_child(self):
        x = self.parent / self.message_to_parent
        weight = self.weight.value()
        bias = self.bias.value()
        m = tf.tensordot(x.mean(), weight, 1) + bias
        v = tf.tensordot(x.variance(), weight ** 2, 1)
        message_to_child = GaussianArray.from_array(m, v)
        self.child.update(self.message_to_child, message_to_child)
        self.message_to_child = message_to_child

    def to_parent(self):
        result = self.child / self.message_to_child
        x = self.parent / self.message_to_parent
        weight = self.weight.value()
        bias = self.bias.value()
        m = (tf.expand_dims(result.mean() - bias - tf.tensordot(x.mean(), weight, 1), 1) +
             tf.expand_dims(x.mean(), -1) * tf.expand_dims(weight, 0)) / tf.expand_dims(weight, 0)
        v = (tf.expand_dims(result.variance() + tf.tensordot(x.variance(), weight ** 2, 1), 1) -
             tf.expand_dims(x.variance(), -1) * tf.expand_dims(weight ** 2, 0)) / tf.expand_dims(weight ** 2, 0)
        p = 1.0 / v
        mtp = m * p
        message_to_parent = GaussianArray.from_array_natural(
            tf.reduce_sum(p, -1),
            tf.reduce_sum(mtp, -1)
        )
        self.parent.update(self.message_to_parent, message_to_parent)
        self.message_to_parent = message_to_parent


class Concatenate(VMPFactor):

    def __init__(self, vector: GaussianArray, parts: Dict[str, GaussianArray]):
        super().__init__()
        self._deterministic = True
        self.vector = vector
        shape_in = {n: p.shape() for n, p in parts.items()}
        self.parts = parts
        shape_out = vector.shape()
        d = len(shape_out)
        self.message_to_parts = {k: GaussianArray.uniform(s) for k, s in shape_in.items()}
        self.message_to_vector = GaussianArray.uniform(shape_out)
        size = [s[-1] for k, s in shape_in.items()]
        begin = [0, *np.cumsum(size[:-1])]
        self._size = [tuple([*shape_out[:-1], s]) for s in size]
        self._begin = [tuple([*[0] * (d - 1), s]) for s in begin]
        self._name = [k for k, s in shape_in.items()]

    def to_parts(self):
        v = self.vector / self.message_to_vector
        p, mtp = v.natural()
        for name, begin, size in zip(self._name, self._begin, self._size):
            message_to_part = GaussianArray.from_array_natural(
                tf.slice(p, begin, size),
                tf.slice(mtp, begin, size)
            )
            self.parts[name].update(self.message_to_parts[name], message_to_part)
            self.message_to_parts[name] = message_to_part

    def to_vector(self):
        parts = dict()
        for name in self._name:
            parts[name] = self.parts[name] / self.message_to_parts[name]
        p = tf.concat([part.precision() for k, part in parts.items()], -1)
        mtp = tf.concat([part.mean_times_precision() for k, part in parts.items()], -1)
        message_to_vector = GaussianArray.from_array_natural(p, mtp)
        self.vector.update(self.message_to_vector, message_to_vector)
        self.message_to_vector = message_to_vector


class ExpandTranspose(VMPFactor):
    # special factor for heterogeneity
    # takes a Nx1 tensor and outputs a NxNx2 tensor
    # where the 1 NxN is the original tensor repeated N times
    # and the second NxN is the same but transposed

    def __init__(self, child, parent):
        super().__init__()
        self._deterministic = True
        self.child = child
        self.parent = parent
        self.N = parent.shape()[0]
        self.message_to_child = GaussianArray.uniform((self.N, self.N, 2))
        self.message_to_parent = GaussianArray.uniform((self.N, 1))

    def to_child(self):
        parent = self.parent / self.message_to_parent
        p, mtp = parent.natural()
        p0 = tf.tile(tf.expand_dims(p, 0), [self.N, 1, 1])
        mtp0 = tf.tile(tf.expand_dims(mtp, 0), [self.N, 1, 1])
        p1 = tf.tile(tf.expand_dims(p, 1), [1, self.N, 1])
        mtp1 = tf.tile(tf.expand_dims(mtp, 1), [1, self.N, 1])
        p = tf.concat([p0, p1], 2)
        mtp = tf.concat([mtp0, mtp1], 2)
        message_to_child = GaussianArray(p, mtp)
        self.child.update(self.message_to_child, message_to_child)
        self.message_to_child = message_to_child

    def to_parent(self):
        child = self.child / self.message_to_child
        p, mtp = child.natural()
        p0 = tf.slice(p, [0, 0, 0], [-1, -1, 1])
        mtp0 = tf.slice(mtp, [0, 0, 0], [-1, -1, 1])
        p1 = tf.slice(p, [0, 0, 1], [-1, -1, 1])
        mtp1 = tf.slice(mtp, [0, 0, 1], [-1, -1, 1])
        p = tf.reduce_sum(p0, 0) + tf.reduce_sum(p1, 1)
        mtp = tf.reduce_sum(mtp0, 0) + tf.reduce_sum(mtp1, 1)
        message_to_parent = GaussianArray(p, mtp)
        self.parent.update(self.message_to_parent, message_to_parent)
        self.message_to_parent = message_to_parent


# TODO: Logit
