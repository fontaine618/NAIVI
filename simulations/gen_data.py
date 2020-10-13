import tensorflow as tf
import numpy as np


def generate_dataset(
        N, K, p_cts, p_bin, var_adj=1., var_cov=1., missing_rate=0.2, alpha_mean=-1.,
        seed=1, link_model="Logistic", bin_model="Logistic"
):
    tf.random.set_seed(seed)
    p = p_cts + p_bin
    # -------------------------------------------------
    # latent variables
    Z = tf.random.normal((N, K), 0.0, 1.0, tf.float32)
    alpha = tf.random.normal((N, 1), alpha_mean, 0.5, tf.float32)
    # regression matrix
    B = tf.random.normal((K, p))
    B = B / tf.norm(B, 2, 0, keepdims=True)
    B0 = 0. * tf.ones((1, p))
    # covariate model
    Theta_X = tf.matmul(Z, B) + B0
    # continuous variables
    X_cts_all = tf.slice(Theta_X, [0, 0], [-1, p_cts])
    X_cts_all += tf.random.normal((N, p_cts), 0.0, var_cov, tf.float32)
    missing = tf.random.uniform(X_cts_all.shape) < missing_rate
    X_cts = tf.where(missing, np.nan, X_cts_all)
    X_cts_missing = tf.where(missing, X_cts_all, np.nan)
    if p_cts == 0:
        X_cts_missing = None
    # binary covariates
    X_bin_all = tf.slice(Theta_X, [0, p_cts], [-1, -1])
    if bin_model == "NoisyProbit":
        X_bin_all += tf.random.normal((N, p_bin), 0.0, var_cov, tf.float32)
        X_bin_all = tf.where(X_bin_all > 0., 1., 0.)
    else:
        P = 1. / (1. + tf.math.exp(- X_bin_all))
        X_bin_all = tf.where(tf.random.uniform(P.shape) < P, 1., 0.)
    missing = tf.random.uniform(X_bin_all.shape) < missing_rate
    X_bin = tf.where(missing, np.nan, X_bin_all)
    X_bin_missing = tf.where(missing, X_bin_all, np.nan)
    if p_bin == 0:
        X_bin_missing = None
    # adjacency model
    Theta_A = alpha + tf.transpose(alpha) + tf.matmul(Z, Z, transpose_b=True)
    if link_model == "NoisyProbit":
        A = tf.where(Theta_A + tf.random.normal((N, N), 0., var_adj) > 0., 1., 0.)
    else:
        P = 1. / (1. + tf.math.exp(- Theta_A))
        A = tf.where(tf.random.uniform(P.shape) < P, 1., 0.)
    # only upper triangular part
    lower = tf.ones_like(A)
    upper = tf.linalg.band_part(lower, -1, 0) == 0
    A = tf.where(upper, A, np.nan)
    # return
    return Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, A, B, B0

