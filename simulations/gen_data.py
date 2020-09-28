import tensorflow as tf
import numpy as np


def generate_dataset(
        N, K, p_cts, p_bin, var_adj=1., var_cov=1., missing_rate=0.2,
        seed=1

):
    tf.random.set_seed(seed)
    p = p_cts + p_bin
    # -------------------------------------------------
    # latent variables
    Z = tf.random.normal((N, K), 0.0, 1.0, tf.float32)
    alpha = tf.random.normal((N, 1), 0.0, 1.0, tf.float32)
    # regression matrix
    B = tf.ones((K, p)) * 2.
    B0 = 1. * tf.ones((1, p))
    # covariate model
    Theta_X = tf.matmul(Z, B) + B0
    # continuous variables
    X_cts_all = tf.slice(Theta_X, [0, 0], [-1, p_cts])
    X_cts_all += tf.random.normal((N, p_cts), 0.0, var_cov, tf.float32)
    X_bin_all = tf.slice(Theta_X, [0, p_cts], [-1, -1])
    X_bin_all += tf.random.normal((N, p_bin), 0.0, var_cov, tf.float32)
    X_bin_all = tf.where(X_bin_all > 0., 1., 0.)
    # # missing values
    missing = tf.random.uniform(X_cts_all.shape) < missing_rate
    X_cts = tf.where(missing, np.nan, X_cts_all)
    missing = tf.random.uniform(X_bin_all.shape) < missing_rate
    X_bin = tf.where(missing, np.nan, X_bin_all)
    # adjacency model
    Theta_A = alpha + tf.transpose(alpha) + tf.matmul(Z, Z, transpose_b=True)
    A = tf.where(Theta_A + tf.random.normal((N, N), 0., var_adj) > 0., 1., 0.)
    lower = tf.ones_like(A)
    upper = tf.linalg.band_part(lower, -1, 0) == 0
    A_lower = tf.where(upper, A, np.nan)
    return Z, alpha, X_cts, X_bin, A_lower

