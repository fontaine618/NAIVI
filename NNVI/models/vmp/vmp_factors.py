import tensorflow as tf
import numpy as np
from NNVI.models.gaussianarray import GaussianArray
from NNVI.models.bernoulliarray import BernoulliArray
import tensorflow_probability as tfp


class VMPFactor:

    def __init__(self):
        self._deterministic = True

    def to_elbo(self, **kwargs):
        if self._deterministic:
            # if not defined in child, we assume deterministic node
            # and the contribution is 0
            return 0.
        else:
            # otherwise it must be implemented in the child class
            raise NotImplementedError("to_elbo not implemented for this stochastic factor")


class Prior(VMPFactor):
    # TODO: merge with other two Gaussian factors

    def __init__(self, prior):
        super().__init__()
        self._deterministic = False
        self._prior = prior
        self.message_to_x = prior

    def to_x(self):
        return self._prior

    def to_elbo(self, x):
        elbo = self.message_to_x.variance() + self.message_to_x.mean() ** 2
        elbo += x.variance() + x.mean() ** 2
        elbo += -2. * x.mean() * self.message_to_x.mean()
        elbo /= self.message_to_x.variance()
        # elbo += tf.math.log(2 * np.pi)
        elbo += tf.math.log(self.message_to_x.variance())
        return -.5 * tf.reduce_sum(elbo)


class GaussianComparison(VMPFactor):
    # this is basically the same as prior but the other way around and with a different structure
    # maybe at some point it would be good to merge them
    # mean Nxp
    # variance 1xp
    # N(mean, variance) where variance is constant per column
    # TODO: put variance as an attribute

    def __init__(self, shape):
        super().__init__()
        self._deterministic = False
        self.message_to_mean = GaussianArray.uniform(shape)
        self.message_to_x = GaussianArray.uniform(shape)

    def to_mean(self, x, variance):
        m = x.mean()
        v = variance * tf.ones_like(m) + x.variance_safe() # maybe this contains infinity
        self.message_to_mean = GaussianArray.from_array(m, v)
        return self.message_to_mean

    def to_x(self, mean, variance):
        mean = mean / self.message_to_mean
        m = mean.mean()
        v = mean.variance() + variance
        self.message_to_x = GaussianArray.from_array(m, v)
        return self.message_to_x

    def to_elbo(self, mean, x, variance):
        v = variance * tf.ones(x.shape())
        elbo = mean.variance() + mean.mean() ** 2
        elbo += x.variance_safe() + x.mean() ** 2
        elbo += -2. * x.mean() * mean.mean()
        elbo /= v
        # elbo += tf.math.log(2 * np.pi)
        elbo += tf.math.log(v)
        # remove unobserved values
        # TODO: make safe for gradient when missing values
        return -.5 * tf.reduce_sum(tf.where(x.is_uniform(), 0.0, elbo))


class Sum(VMPFactor):

    # sum over last dimension (could be expanded to and dim by replacing -1 <- dim with argument)

    def __init__(self, shape_in, shape_out):
        super().__init__()
        self.message_to_sum = GaussianArray.uniform(shape_out)
        self.message_to_x = GaussianArray.uniform(shape_in)

    def to_sum(self, x):
        x = x / self.message_to_x
        self.message_to_sum = GaussianArray.from_array(
            tf.math.reduce_sum(x.mean(), -1),
            tf.math.reduce_sum(x.variance(), -1)
        )
        return self.message_to_sum

    def to_x(self, x, sum):
        sum = sum / self.message_to_sum
        x = x / self.message_to_x
        m = tf.expand_dims(sum.mean(), -1) - tf.math.reduce_sum(x.mean(), -1, keepdims=True) + x.mean()
        v = tf.where(
            x.is_uniform(),
            np.inf,
            tf.expand_dims(sum.variance(), -1) + tf.math.reduce_sum(x.variance(), -1, keepdims=True) - x.variance()
        )
        self.message_to_x = GaussianArray.from_array(m, v)
        return self.message_to_x


class Product(VMPFactor):
    # all pairwise products in the first dimension (N, ...) => (N, N, ...)

    def __init__(self, shape_in, shape_out):
        super().__init__()
        self.message_to_x = GaussianArray.uniform(shape_in)
        self.message_to_product = GaussianArray.uniform(shape_out)

    def to_product(self, x):
        x = x / self.message_to_x
        m0 = tf.expand_dims(x.mean(), 0)
        m1 = tf.expand_dims(x.mean(), 1)
        v0 = tf.expand_dims(x.variance(), 0)
        v1 = tf.expand_dims(x.variance(), 1)
        m = m0 * m1
        v = m0 ** 2 * v1 + m1 ** 2 * v0 + v0 * v1
        self.message_to_product = GaussianArray.from_array(m, v)
        return self.message_to_product

    def to_x(self, product, x):
        product = product / self.message_to_product
        x = x / self.message_to_x
        # here x contains the message from the other term in the product
        # message to x0 using x as the message from c1
        p = product.precision() * tf.expand_dims(x.variance() + x.mean() ** 2, 1)
        mtp = product.mean_times_precision() * tf.expand_dims(x.mean(), 1)
        p0 = tf.math.reduce_sum(p, 1)
        mtp0 = tf.math.reduce_sum(mtp, 1)
        # message to x1 using x as the message from x0
        p = product.precision() * tf.expand_dims(x.variance() + x.mean() ** 2, 0)
        mtp = product.mean_times_precision() * tf.expand_dims(x.mean(), 0)
        p1 = tf.math.reduce_sum(p, 0)
        mtp1 = tf.math.reduce_sum(mtp, 0)
        # product of messages
        self.message_to_x = GaussianArray(p0 + p1, mtp0 + mtp1)
        return self.message_to_x


class WeightedSum(VMPFactor):

    # expects weighted sum over last dimension
    # x is NxK, B is KxP and B0 is 1xP
    # result is NxP

    def __init__(self, shape_in, shape_out):
        super().__init__()
        self.message_to_x = GaussianArray.uniform(shape_in)
        self.message_to_result = GaussianArray.uniform(shape_out)

    def to_result(self, x, weight, bias):
        x = x / self.message_to_x
        m = tf.tensordot(x.mean(), weight, 1) + bias
        v = tf.tensordot(x.variance(), weight ** 2, 1)
        self.message_to_result = GaussianArray.from_array(m, v)
        return self.message_to_result

    def to_x(self, x, result, weight, bias):
        result = result / self.message_to_result
        x = x / self.message_to_x
        m = (tf.expand_dims(result.mean() - bias - tf.tensordot(x.mean(), weight, 1), 1) +
             tf.expand_dims(x.mean(), -1) * tf.expand_dims(weight, 0)) / tf.expand_dims(weight, 0)
        v = (tf.expand_dims(result.variance() + tf.tensordot(x.variance(), weight ** 2, 1), 1) -
             tf.expand_dims(x.variance(), -1) * tf.expand_dims(weight ** 2, 0)) / tf.expand_dims(weight ** 2, 0)
        p = 1.0 / v
        mtp = m * p
        self.message_to_x = GaussianArray.from_array_natural(
            tf.reduce_sum(p, -1),
            tf.reduce_sum(mtp, -1)
        )
        return self.message_to_x


class AddVariance(VMPFactor):
    # TODO: this is identical to GaussianComparison
    # Stochastic node: N(Gaussian mean, fixed variance)
    # i.e. increases the variance

    def __init__(self, shape):
        super().__init__()
        self._deterministic = False
        self.message_to_x = GaussianArray.uniform(shape)
        self.message_to_mean = GaussianArray.uniform(shape)

    def to_x(self, mean, variance):
        mean = mean / self.message_to_mean
        m = mean.mean()
        v = mean.variance() + variance
        self.message_to_x = GaussianArray.from_array(m, v)
        return self.message_to_x

    def to_mean(self, x, variance):
        x = x / self.message_to_x
        m = x.mean()
        v = x.variance() + variance
        self.message_to_mean = GaussianArray.from_array(m, v)
        return self.message_to_mean

    def to_elbo(self, mean, x, variance):
        variance = variance * tf.ones_like(x.precision())

        elbo = mean.variance() + mean.mean() ** 2
        elbo += x.variance() + x.mean() ** 2  # first term should always be 0.?
        elbo += -2. * x.mean() * mean.mean()
        elbo /= variance
        # elbo += tf.math.log(2 * np.pi)
        elbo += tf.math.log(variance)
        return -.5 * tf.reduce_sum(elbo)


class Probit(VMPFactor):
    # TODO incorporate the variance directly, move elbo here

    def __init__(self, shape):
        super().__init__()
        self.message_to_x = GaussianArray.uniform(shape)
        self.message_to_result = BernoulliArray.uniform(shape)

    def to_x(self, x, A):
        # TODO: assumes 0/1 only, need to fix later with missing values
        x = x / self.message_to_x
        A = A.proba()
        stnr = x.mean() * tf.math.sqrt(x.precision()) * tf.cast(2*A - 1, tf.float32)
        vf = tfp.distributions.Normal(0., 1.).prob(stnr) / tfp.distributions.Normal(0., 1.).cdf(stnr)
        wf = vf * (stnr + vf)
        m = x.mean() + tf.math.sqrt(x.variance()) * vf * tf.cast(2*A - 1, tf.float32)
        v = x.variance() * (1. - wf)
        self.message_to_x = GaussianArray.from_array(m, v)
        return self.message_to_x

    def to_result(self, x):
        x = x / self.message_to_x
        proba = 1. - tfp.distributions.Normal(*x.mean_and_variance()).cdf(0.0)
        return BernoulliArray.from_array(proba)


class ProbitNoisy(VMPFactor):

    def __init__(self, shape, variance):
        super().__init__()
        self._deterministic = False
        self.message_to_x = GaussianArray.uniform(shape)
        self.message_to_result = BernoulliArray.uniform(shape)
        self.variance = variance

    def to_x(self, x, A):
        # compute incoming message
        from_x = x / self.message_to_x
        # add noise
        from_x = GaussianArray.from_array(
            mean=from_x.mean(),
            variance=from_x.variance() + self.variance.value()
        )
        # compute outgoing message
        A = A.proba()  # TODO this assumes 0/1, need to fix later
        stnr = from_x.mean() * tf.math.sqrt(from_x.precision()) * tf.cast(2*A - 1, tf.float32)
        vf = tfp.distributions.Normal(0., 1.).prob(stnr) / tfp.distributions.Normal(0., 1.).cdf(stnr)
        wf = vf * (stnr + vf)
        m = from_x.mean() + tf.math.sqrt(from_x.variance()) * vf * tf.cast(2*A - 1, tf.float32)
        v = from_x.variance() * (1. - wf)
        # add noise
        v += self.variance.value()
        self.message_to_x = GaussianArray.from_array(m, v)
        return self.message_to_x

    def to_result(self, x):
        # compute incoming message
        from_x = x / self.message_to_x
        # add noise
        from_x = GaussianArray.from_array(
            mean=from_x.mean(),
            variance=from_x.variance() + self.variance.value()
        )
        # compute probability
        proba = 1. - tfp.distributions.Normal(*from_x.mean_and_variance()).cdf(0.0)
        return BernoulliArray.from_array(proba)

    def to_elbo(self, x, A):
        mean = x
        x = GaussianArray.from_array(
            mean=x.mean(),
            variance=x.variance() + self.variance.value()
        )
        elbo = mean.variance() + mean.mean() ** 2
        elbo += x.variance() + x.mean() ** 2
        elbo += -2. * x.mean() * mean.mean()
        elbo /= self.variance.value()
        # elbo += tf.math.log(2 * np.pi)
        elbo += tf.math.log(self.variance.value())
        return -.5 * tf.reduce_sum(elbo)


# TODO: Logit

class Concatenate(VMPFactor):

    def __init__(self, shape_in, shape_out):
        super().__init__()
        d = len(shape_out)
        self.message_to_x = {k: GaussianArray.uniform(s) for k, s in shape_in.items()}
        self.message_to_v = GaussianArray.uniform(shape_out)
        size = [s[-1] for k, s in shape_in.items()]
        begin = [0, *np.cumsum(size[:-1])]
        self._size = [tuple([*shape_out[:-1], s]) for s in size]
        self._begin = [tuple([*[0]*(d - 1), s]) for s in begin]
        self._name = [k for k, s in shape_in.items()]

    def to_x(self, v):
        v = v / self.message_to_v
        p, mtp = v.natural()
        for name, begin, size in zip(self._name, self._begin, self._size):
            self.message_to_x[name] = GaussianArray.from_array_natural(
                tf.slice(p, begin, size),
                tf.slice(mtp, begin, size)
            )
        return self.message_to_x

    def to_v(self, x):
        for k in x.keys():
            x[k] = x[k] / self.message_to_x[k]
        p = tf.concat([xx.precision() for k, xx in x.items()], -1)
        mtp = tf.concat([xx.mean_times_precision() for k, xx in x.items()], -1)
        self.message_to_v = GaussianArray.from_array_natural(p, mtp)
        return self.message_to_v

