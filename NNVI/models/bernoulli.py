import tensorflow as tf
import numpy as np
from .distribution import Distribution


class BernoulliArray(Distribution):

    def __init__(self, proba):
        super().__init__()
        self._proba = proba

    def is_point_mass(self):
        return tf.math.logical_or(self._proba == 0., self._proba == 1.)

    def is_uniform(self):
        return self._proba == 0.5

    def mean(self):
        return self.proba

    def variance(self):
        return self.proba * (1. - self.proba)

    def precision(self):
        return 1. / self.variance()

    def mean_times_precision(self):
        return self.mean() * self.precision()

    def natural(self):
        p = self.precision()
        return p, self.mean() * p

    def mean_and_variance(self):
        return self.mean(), self.variance()

    def mode(self):
        return tf.where(self._proba > 0.5, 1., 0.)

    def shape(self):
        return self._proba.shape

    def entropy(self):
        p = self._proba
        entropy = p * tf.math.log(p) + (1. - p) * tf.math.log(1. - p)
        return tf.reduce_sum(tf.where(tf.math.is_nan(entropy), 0., entropy))

    def logit(self):
        p = self._proba
        return tf.math.log(p) - tf.math.log(1. - p)

    @classmethod
    def from_array(cls, proba):
        return cls(proba)

    @classmethod
    def from_array_logit(cls, logit):
        e = - tf.math.exp(logit)
        return cls(1. / (1. + e))

    @classmethod
    def uniform(cls, shape):
        return cls.from_array(0.5 * tf.ones(shape))

    @classmethod
    def observed(cls, point):
        m = point
        v = tf.zeros_like(m)
        return cls.from_array(m, v)

    def __str__(self):
        out = "BernoulliArray{}\n".format(self.shape())
        out += "Probabilities=\n" + str(self._proba)
        return out

