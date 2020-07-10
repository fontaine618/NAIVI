import tensorflow as tf
from NNVI.models.gaussian import GaussianArray
from NNVI.models.parameter import ParameterArray
from NNVI.models.vmp.vmp_factors import Prior, Product, Probit, Sum, AddVariance, \
    Concatenate, WeightedSum, GaussianComparison
import tensorflow_probability as tfp


class JointModel:

    def __init__(self, N, K, A, X):
        self.A = A
        self.X = X
        self.N = N
        self.K = K
        self.p = X.shape[1]
        self.parameters = {
            "noise_adjacency": ParameterArray(1. * tf.ones((1, 1))),
            "B": ParameterArray(tf.ones((self.K, self.p))),
            "B0": ParameterArray(tf.ones((1, self.p))),
            "noise_covariate": ParameterArray(tf.ones((1, self.p)))
        }
        self.nodes = {
            "latent": GaussianArray.uniform((N, K)),
            "heterogeneity": GaussianArray.uniform((N, 1)),
            "product": GaussianArray.uniform((N, N, K)),
            "vector": GaussianArray.uniform((N, N, K+2)),
            "linear_predictor_adjacency": GaussianArray.uniform((N, N)),
            "noisy_linear_predictor_adjacency": GaussianArray.uniform((N, N)),
            "links": tf.zeros((N, N)),
            "linear_predictor_covariate": GaussianArray.uniform((N, self.p))
        }
        self.factors = {
            "latent_prior": Prior(GaussianArray.from_shape((N, K), 0., 1.)),
            "heterogeneity_prior": Prior(GaussianArray.from_shape((N, 1), 0., 1.)),
            "product": Product((N, K), (N, N, K)),
            "concatenate": Concatenate({"a_u": (N, N, 1), "a_v": (N, N, 1), "s_uv": (N, N, K)}, (N, N, K+2)),
            "sum": Sum((N, N, K+2), (N, N)),
            "noise": AddVariance((N, N)),
            "adjacency": Probit((N, N)),
            "weighted_sum": WeightedSum((N, K), (N, self.p)),
            "comparison_gaussian": GaussianComparison((N, self.p))
        }

    def _break_symmetry(self):
        self.factors["latent_prior"].message_to_x = GaussianArray.from_array(
                        mean=tf.random.normal((self.N, self.K), 0., 1.),
                        variance=tf.ones((self.N, self.K)) * 1000.
                    )

    def initialize_latent(self):
        self.nodes["latent"] = self.factors["latent_prior"].to_x() * \
            self.factors["product"].message_to_x * \
            self.factors["weighted_sum"].message_to_x

    def forward_adjacency(self):
        # product
        self.nodes["product"] = self.factors["product"].to_product(
            x=self.nodes["latent"]
        ) * self.factors["concatenate"].message_to_x["s_uv"]
        # heterogeneity
        to_alpha = \
            self.factors["concatenate"].message_to_x["a_u"].product(0) * \
            self.factors["concatenate"].message_to_x["a_v"].product(1)
        self.nodes["heterogeneity"] = self.factors["heterogeneity_prior"].message_to_x * to_alpha
        # concatenate
        alpha = self.nodes["heterogeneity"]
        x = {
            "a_u": GaussianArray(
                        tf.tile(tf.expand_dims(alpha.precision(), 0), [self.N, 1, 1]),
                        tf.tile(tf.expand_dims(alpha.mean_times_precision(), 0), [self.N, 1, 1])
                    ),
            "a_v": GaussianArray(
                        tf.tile(tf.expand_dims(alpha.precision(), 1), [1, self.N, 1]),
                        tf.tile(tf.expand_dims(alpha.mean_times_precision(), 1), [1, self.N, 1])
                    ),
            "s_uv": self.nodes["product"]
        }
        self.nodes["vector"] = self.factors["concatenate"].to_v(x) * \
            self.factors["sum"].to_x(self.nodes["vector"], self.nodes["linear_predictor_adjacency"])
        # linear predictor
        self.nodes["linear_predictor_adjacency"] = \
            self.factors["sum"].to_sum(self.nodes["vector"]) * \
            self.factors["noise"].message_to_mean
        # noizy linear predictor
        self.nodes["noisy_linear_predictor_adjacency"] = self.factors["noise"].to_x(
            mean=self.nodes["linear_predictor_adjacency"],
            variance=self.parameters["noise_adjacency"].value
        ) * self.factors["adjacency"].message_to_x

    def backward_adjacency(self):
        # noisy linear predictors
        self.nodes["noisy_linear_predictor_adjacency"] = \
            self.factors["adjacency"].to_x(self.nodes["noisy_linear_predictor_adjacency"], self.A) * \
            self.factors["noise"].message_to_x
        # linear predictors
        self.nodes["linear_predictor_adjacency"] = \
            self.factors["noise"].to_mean(
                self.nodes["noisy_linear_predictor_adjacency"],
                self.parameters["noise_adjacency"].value
            ) * self.factors["sum"].message_to_sum
        # sum
        self.nodes["vector"] = \
            self.factors["sum"].to_x(self.nodes["vector"], self.nodes["linear_predictor_adjacency"]) * \
            self.factors["concatenate"].message_to_v
        # split
        to_x = self.factors["concatenate"].to_x(self.nodes["vector"])
        to_alpha = to_x["a_u"].product(0) * to_x["a_v"].product(1)
        to_product = to_x["s_uv"]
        # heterogeneity
        self.nodes["heterogeneity"] = to_alpha * self.factors["heterogeneity_prior"].message_to_x
        # latent
        self.nodes["product"] = to_product * self.factors["product"].message_to_product
        self.nodes["latent"] = \
            self.nodes["latent"] / \
            self.factors["product"].message_to_x * \
            self.factors["product"].to_x(self.nodes["product"], self.nodes["latent"])

    def forward_covariate(self):
        self.nodes["linear_predictor_covariate"] = \
            self.factors["weighted_sum"].to_result(
                x=self.nodes["latent"],
                B=self.parameters["B"].value,
                B0=self.parameters["B0"].value
            ) * self.factors["comparison_gaussian"].message_to_mean

    def backward_covariate(self):
        self.nodes["linear_predictor_covariate"] = \
            self.factors["comparison_gaussian"].to_mean(
                X=self.X,
                variance=self.parameters["noise_covariate"].value
            ) * self.factors["weighted_sum"].message_to_result
        self.nodes["latent"] = \
            self.nodes["latent"] / \
            self.factors["weighted_sum"].message_to_x * \
            self.factors["weighted_sum"].to_x(
                x=self.nodes["latent"],
                result=self.nodes["linear_predictor_covariate"],
                B=self.parameters["B"].value,
                B0=self.parameters["B0"].value
            )
