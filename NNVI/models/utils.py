import tensorflow as tf


def invariant_matrix_distance(y_true, y_pred):
    ip_true = tf.linalg.matmul(y_true, y_true, transpose_b=True)
    ip_pred = tf.linalg.matmul(y_pred, y_pred, transpose_b=True)
    diff = ip_true - ip_pred
    num = tf.reduce_sum(diff ** 2)
    denum = tf.reduce_sum(ip_true ** 2)
    return num / denum


def projection_distance(y_true, y_pred):
    inv_true = tf.linalg.matmul(y_true, y_true, transpose_a=True)
    inv_true = tf.linalg.inv(inv_true)
    proj_true = tf.matmul(y_true, inv_true)
    proj_true = tf.matmul(proj_true, y_true, transpose_b=True)
    inv_pred = tf.linalg.matmul(y_pred, y_pred, transpose_a=True)
    inv_pred = tf.linalg.inv(inv_pred)
    proj_pred = tf.matmul(y_pred, inv_pred)
    proj_pred = tf.matmul(proj_pred, y_pred, transpose_b=True)
    return tf.reduce_sum((proj_pred - proj_true) ** 2)