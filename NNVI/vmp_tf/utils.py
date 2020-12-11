import tensorflow as tf


def invariant_matrix_distance(y_true, y_pred):
    ip_true = tf.linalg.matmul(y_true, y_true, transpose_b=True)
    ip_pred = tf.linalg.matmul(y_pred, y_pred, transpose_b=True)
    diff = ip_true - ip_pred
    num = tf.reduce_sum(diff ** 2)
    denum = tf.reduce_sum(ip_true ** 2)
    return num / denum


def projection_distance(y_true, y_pred):
    _, u, _ = tf.linalg.svd(y_true)
    proj_true = tf.matmul(u, u, transpose_b=True)
    _, u, _ = tf.linalg.svd(y_pred)
    proj_pred = tf.matmul(u, u, transpose_b=True)
    return tf.reduce_sum((proj_pred - proj_true) ** 2)