import tensorflow as tf
import numpy as np
from .distributionarray import DistributionArray


class GaussianArray(DistributionArray):

    def __init__(self, precision, mean_times_precision):
        super().__init__()
        self._precision = precision
        self._mean_times_precision = mean_times_precision

    def is_point_mass(self):
        return tf.math.is_inf(self.precision())

    def is_uniform(self):
        return tf.math.logical_and(self.precision() == 0.0, self.mean_times_precision() == 0.0)

    def precision(self):
        return self._precision

    def mean_times_precision(self):
        return self._mean_times_precision

    def natural(self):
        return self._precision, self._mean_times_precision

    def mean(self):
        return tf.where(self._precision < np.inf,
                        # if not point mass, go regular computation
                        tf.where(
                            self._precision > 0.0,
                            # if not uniform, do regular computation
                            self._mean_times_precision / self._precision,
                            # otherwise, return 0
                            0.0
                        ),
                        # otherwise, mtp contains the point
                        self._mean_times_precision)

    def variance(self):
        return tf.where(self._precision > 0.0, 1.0 / self._precision, np.inf)

    def precision_safe(self):
        return tf.clip_by_value(self._precision, 0., 1.0e10)

    def variance_safe(self):
        return tf.clip_by_value(1.0 / self._precision, 0., 1.0e10)

    def mean_and_variance(self):
        return self.mean(), self.variance()

    def mean_and_stddev(self):
        return self.mean(), tf.math.sqrt(self.variance())

    def mode(self):
        return self.mean()

    def shape(self):
        return self._precision.shape

    def product(self, dim):
        return GaussianArray(
            tf.reduce_sum(self._precision, dim),
            tf.reduce_sum(self._mean_times_precision, dim)
        )

    def set_natural(self, p, mtp):
        self._precision = p
        self._mean_times_precision = mtp

    def set_mean_and_variance(self, m, v):
        p = 1. / v
        mtp = m * p
        self.set_natural(p, mtp)

    def update(self, previous, new):
        self.set_to(self * new / previous)
        # print(self.shape())
        # print(tf.reduce_max(tf.math.abs(self.mean())))

    def set_to(self, x):
        self._precision = x.precision()
        self._mean_times_precision = x.mean_times_precision()

    def entropy(self):
        entropy = 0.5 * tf.math.log(self.variance())
        return tf.reduce_sum(tf.where(tf.math.logical_or(self.is_point_mass(), self.is_uniform()), 0., entropy))

    @classmethod
    def from_array(cls, mean, variance):
        p = 1.0 / variance
        mtp = tf.where(p < np.inf, mean * p, mean)
        return cls(p, mtp)

    @classmethod
    def from_array_natural(cls, precision, mean_times_precision):
        return cls(precision, mean_times_precision)

    @classmethod
    def from_shape(cls, shape, mean, variance):
        m = tf.ones(shape, dtype=tf.dtypes.float32) * mean
        v = tf.ones(shape, dtype=tf.dtypes.float32) * variance
        return cls.from_array(m, v)

    @classmethod
    def from_shape_natural(cls, shape, precision, mean_times_precision):
        p = tf.ones(shape, dtype=tf.dtypes.float32) * precision
        mtp = tf.ones(shape, dtype=tf.dtypes.float32) * mean_times_precision
        return cls.from_array_natural(p, mtp)

    @classmethod
    def uniform(cls, shape):
        return cls.from_shape_natural(shape, 0.0, 0.0)

    @classmethod
    def observed(cls, point):
        m = tf.where(tf.math.is_nan(point), 0.0, point)
        v = tf.where(tf.math.is_nan(point), np.inf, 1.0e-10)
        return cls.from_array(m, v)

    def __mul__(self, other):
        if not isinstance(other, GaussianArray):
            raise TypeError("other should be a GaussianArray")
        p = self.precision() + other.precision()
        mtp = self.mean_times_precision() + other.mean_times_precision()
        # case where one of the two is a point mass in which case the product is just the point
        # the case where both are point masses is omitted
        mtp = tf.where(
            tf.math.is_inf(p),
            tf.where(self.is_point_mass(), self.mean(), other.mean()),
            mtp
        )
        return GaussianArray(p, mtp)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if not isinstance(other, GaussianArray):
            raise TypeError("other should be a GaussianArray")
        p, mtp = self.natural()
        op, omtp = other.natural()
        mask = other.is_uniform()
        pnew = tf.where(mask, p, p - op)
        mtpnew = tf.where(mask, mtp, mtp - omtp)
        # case where self is a point mass, in which case division does nothing (unless other is also a point mass, but
        # we omit that case as it should never happen)
        mtpnew = tf.where(self.is_point_mass(), mtp, mtpnew)
        return GaussianArray(pnew, mtpnew)

    def __str__(self):
        out = "GaussianArray{}\n".format(self.shape())
        out += "Mean=\n" + str(self.mean()) + "\n"
        out += "Variance=\n" + str(self.variance())
        return out

