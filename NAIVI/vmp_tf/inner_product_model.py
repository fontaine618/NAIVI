import tensorflow as tf
from models.distributions.gaussianarray import GaussianArray
from NAIVI.vmp_tf.vmp.vmp_factors import Product, Concatenate, Sum


class InnerProductModel:

    def __init__(self, N, K):
        self.N = N
        self.K = K
        # incoming messages
        self.from_position = GaussianArray.uniform((N, K))
        self.from_heterogeneity = GaussianArray.uniform((N, 1))
        self.from_adjacency = GaussianArray.uniform((N, N))
        # nodes
        self.products = GaussianArray.uniform((N, N, K))
        self.vectors = GaussianArray.uniform((N, N, K+2))
        self.linear_predictors = GaussianArray.uniform((N, N))
        # factors
        self.product = Product((N, K), (N, N, K))
        self.concat = Concatenate({"a_u": (N, N, 1), "a_v": (N, N, 1), "s_uv": (N, N, K)}, (N, N, K+2))
        self.sum = Sum((N, N, K+2), (N, N))

    def forward(self, position, heterogeneity, from_adjacency):
        # update products
        self.products = self.product.to_product(position) * self.concat.message_to_x["s_uv"]
        # update vector
        x = {
            "a_u": GaussianArray(
                tf.tile(tf.expand_dims(heterogeneity.precision(), 0), [self.N, 1, 1]),
                tf.tile(tf.expand_dims(heterogeneity.mean_times_precision(), 0), [self.N, 1, 1])
            ),
            "a_v": GaussianArray(
                tf.tile(tf.expand_dims(heterogeneity.precision(), 1), [1, self.N, 1]),
                tf.tile(tf.expand_dims(heterogeneity.mean_times_precision(), 1), [1, self.N, 1])
            ),
            "s_uv": self.products
        }
        self.vectors = self.concat.to_v(x) * self.sum.message_to_x
        # update linear predictors
        self.linear_predictors = self.sum.to_sum(self.vectors) * from_adjacency

    def backward(self, from_adjacency):
        pass

    def elbo(self):
        return 0.