import tensorflow as tf
from models.distributions.gaussianarray import GaussianArray
from NNVI.models.vmp.vmp_factors import Prior


class PositionPrior:

    def __init__(self, N, K, mean=0., variance=1.):
        self.N = N
        self.K = K
        self.mean = mean
        self.variance = variance
        self.prior = Prior(GaussianArray.from_shape((N, K), mean, variance))
        self.marginal = GaussianArray.uniform((N, K))
        self.from_adjacency = GaussianArray.uniform((N, K))
        self.from_covariate = GaussianArray.uniform((N, K))

    def initialize(self):
        # TODO: Check this; I think it does nothing.
        self.prior.message_to_x = GaussianArray.from_array(
                        mean=tf.random.normal((self.N, self.K), self.mean, self.variance),
                        variance=tf.ones((self.N, self.K)) * self.variance
                    )
        self.marginal = self.prior.to_x()

    def update_from_adjacency(self, from_adjacency):
        self.marginal = self.marginal * from_adjacency / self.from_adjacency

    def update_from_covariate(self, from_covariate):
        self.marginal = self.marginal * from_covariate / self.from_covariate

    def elbo(self):
        return 0.


class HeterogeneityPrior:

    def __init__(self, N, mean=0., variance=1.):
        self.N = N
        self.mean = mean
        self.variance = variance
        self.prior = Prior(GaussianArray.from_shape((N, 1), mean, variance))
        self.marginal = GaussianArray.uniform((N, 1))
        self.from_adjacency = GaussianArray.uniform((N, 1))

    def initialize(self):
        self.prior.message_to_x = GaussianArray.from_array(
                        mean=tf.random.normal((self.N, 1), self.mean, self.variance),
                        variance=tf.ones((self.N, 1)) * self.variance
                    )
        self.marginal = self.prior.to_x()

    def update_from_adjacency(self, from_adjacency):
        self.marginal = self.marginal * from_adjacency / self.from_adjacency

    def elbo(self):
        return 0.