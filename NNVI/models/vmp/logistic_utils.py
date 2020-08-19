import tensorflow as tf
import tensorflow_probability as tfp


def sigmoid_integrals(mean, variance, degrees=[0, 1, 2]):
    # integrals of the form
    # int_-inf^inf x^r Phi(m+vx)phi(x) dx
    # for r in degrees

    _p = tf.convert_to_tensor([
        0.003246343272134,
        0.051517477033972,
        0.195077912673858,
        0.315569823632818,
        0.274149576158423,
        0.131076880695470,
        0.027912418727972,
        0.001449567805354
    ], dtype=tf.float32)

    _s = tf.convert_to_tensor([
        1.365340806296348,
        1.059523971016916,
        0.830791313765644,
        0.650732166639391,
        0.508135425366489,
        0.396313345166341,
        0.308904252267995,
        0.238212616409306
    ], dtype=tf.float32)


    n_dim = len(mean.shape)
    for _ in range(n_dim):
        _p = tf.expand_dims(_p, 0)
        _s = tf.expand_dims(_s, 0)

    mean = tf.expand_dims(mean, -1)
    variance = tf.expand_dims(variance, -1)
    t = tf.math.sqrt(1. + _s ** 2 * variance)
    smot = mean * _s / t
    phi = tfp.distributions.Normal(0., 1.).prob(smot)
    Phi = tfp.distributions.Normal(0., 1.).cdf(smot)
    integral = dict()
    if 0 in degrees:
        integral[0] = tf.reduce_sum(_p * Phi, -1)
    if 1 in degrees:
        psot = _p * _s / t
        integral[1] = tf.reduce_sum(psot * phi, -1) * tf.squeeze(tf.math.sqrt(variance), axis=-1)
    if 2 in degrees:
        svot = (_s / t) ** 2 * variance
        integral[2] = tf.reduce_sum(_p * (Phi + svot * smot * phi), -1)
    return integral


def sigmoid_mean_and_variance(mean, variance):
    integrals = sigmoid_integrals(mean, variance)
    sd = tf.math.sqrt(variance)
    sigmoid_mean = mean * integrals[0] + sd * integrals[1]
    sigmoid_mean /= integrals[0]
    sigmoid_variance = mean ** 2 * integrals[0] + 2. * mean * sd * integrals[1] + variance * integrals[2]
    sigmoid_variance /= integrals[0]
    sigmoid_variance -= sigmoid_mean ** 2
    return sigmoid_mean, sigmoid_variance