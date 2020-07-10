import tensorflow as tf


class Model:

    def __init__(self, latent_dim=5):
        self.latent_dim = latent_dim

    def _check_input(self, A, X_continuous=None, X_discrete=None):
        self.n = A.shape[0]
        if not self.n == A.shape[1]:
            raise ValueError("A must be square")
        if X_continuous is not None:
            if not X_continuous.shape[0] == self.n:
                raise ValueError("X_continuous must have the same number of rows as A")
            self.p_continuous = X_continuous.shape[1]
        if X_discrete is not None:
            if not X_discrete.shape[0] == self.n:
                raise ValueError("X_discrete must have the same number of rows as A")
            self.p_discrete = X_discrete.shape[1]
        self.p_discrete = X_discrete.shape[1]
        self.A = A
        self.X_continuous = X_continuous
        self.X_discrete = X_discrete