import tensorflow as tf
import numpy as np
from models.distributions.gaussianarray import GaussianArray
from models.distributions.bernoulliarray import BernoulliArray
from NNVI.models.parameter import ParameterArray, ParameterArrayLogScale
from NNVI.models.vmp.vmp_factors import Prior, Product, Probit, Sum, AddVariance, \
    Concatenate, WeightedSum, GaussianComparison


class JointModel:

    def __init__(self, K, A, X):
        N = A.shape[0]
        p = X.shape[1]
        self.N = N
        self.K = K
        self.p = p
        self.parameters = {
            "noise_adjacency": ParameterArrayLogScale(1. * tf.ones((1, 1))),
            "weight": ParameterArray(tf.ones((K, p))),
            "bias": ParameterArray(0. * tf.ones((1, p))),
            "noise_covariate": ParameterArrayLogScale(tf.ones((1, p)))
        }
        self.nodes = {
            "latent": GaussianArray.uniform((N, K)),
            "heterogeneity": GaussianArray.uniform((N, 1)),
            "product": GaussianArray.uniform((N, N, K)),
            "vector": GaussianArray.uniform((N, N, K+2)),
            "linear_predictor_adjacency": GaussianArray.uniform((N, N)),
            "noisy_linear_predictor_adjacency": GaussianArray.uniform((N, N)),
            "links": BernoulliArray.observed(A),
            "linear_predictor_covariate": GaussianArray.uniform((N, p)),
            "covariates_continuous": GaussianArray.observed(X)
        }
        self.factors = {
            "latent_prior": Prior(GaussianArray.from_shape((N, K), 0., 1.)),
            "heterogeneity_prior": Prior(GaussianArray.from_shape((N, 1), 0., 1.)),
            "product": Product((N, K), (N, N, K)),
            "concatenate": Concatenate({"a_u": (N, N, 1), "a_v": (N, N, 1), "s_uv": (N, N, K)}, (N, N, K+2)),
            "sum": Sum((N, N, K+2), (N, N)),
            "noise": AddVariance((N, N)),
            "adjacency": Probit((N, N)),
            # "adjacency": ProbitNoisy((N, N), self.parameters["noise_adjacency"]),
            "weighted_sum": WeightedSum((N, K), (N, p)),
            "comparison_gaussian": GaussianComparison((N, p))
        }

    def _break_symmetry(self):
        # TODO: I think this should be fixed, it seems to do nothing?
        self.factors["latent_prior"].message_to_x = GaussianArray.from_array(
                        mean=tf.random.normal((self.N, self.K), 0., 1.),
                        variance=tf.ones((self.N, self.K)) * 1.
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
        # heterogeneity (seems unnecessary/out of place?)
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
            self.factors["sum"].message_to_x
            #self.factors["sum"].to_x(self.nodes["vector"], self.nodes["linear_predictor_adjacency"])


        # # linear predictor
        # self.nodes["linear_predictor_adjacency"] = \
        #     self.factors["sum"].to_sum(self.nodes["vector"]) * \
        #     self.factors["adjacency"].message_to_x
        # # adjacency (not necessary, but useful if we need to return a value)
        # return self.factors["adjacency"].to_result(self.nodes["linear_predictor_adjacency"])

        # linear predictor
        self.nodes["linear_predictor_adjacency"] = \
            self.factors["sum"].to_sum(self.nodes["vector"]) * \
            self.factors["noise"].message_to_mean
        # noizy linear predictor
        self.nodes["noisy_linear_predictor_adjacency"] = self.factors["noise"].to_x(
            mean=self.nodes["linear_predictor_adjacency"],
            variance=self.parameters["noise_adjacency"].value()
        ) * self.factors["adjacency"].message_to_x

    def backward_adjacency(self):
        # # adjacency
        # self.nodes["linear_predictor_adjacency"] = \
        #     self.factors["adjacency"].to_x(self.nodes["linear_predictor_adjacency"], self.nodes["links"]) * \
        #     self.factors["sum"].message_to_sum

        # noisy linear predictors
        self.nodes["noisy_linear_predictor_adjacency"] = \
            self.factors["adjacency"].to_x(self.nodes["noisy_linear_predictor_adjacency"], self.nodes["links"]) * \
            self.factors["noise"].message_to_x
        # linear predictors
        self.nodes["linear_predictor_adjacency"] = \
            self.factors["noise"].to_mean(
                self.nodes["noisy_linear_predictor_adjacency"],
                self.parameters["noise_adjacency"].value()
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
                weight=self.parameters["weight"].value(),
                bias=self.parameters["bias"].value()
            ) * self.factors["comparison_gaussian"].message_to_mean

    def backward_covariate(self):
        self.nodes["linear_predictor_covariate"] = \
            self.factors["comparison_gaussian"].to_mean(
                x=self.nodes["covariates_continuous"],
                variance=self.parameters["noise_covariate"].value()
            ) * self.factors["weighted_sum"].message_to_result
        self.nodes["latent"] = \
            self.nodes["latent"] / \
            self.factors["weighted_sum"].message_to_x * \
            self.factors["weighted_sum"].to_x(
                x=self.nodes["latent"],
                result=self.nodes["linear_predictor_covariate"],
                weight=self.parameters["weight"].value(),
                bias=self.parameters["bias"].value()
            )

    def elbo(self):
        elbo = 0.
        elbo += self.factors["latent_prior"].to_elbo(self.nodes["latent"])
        elbo += self.factors["heterogeneity_prior"].to_elbo(self.nodes["heterogeneity"])
        elbo += self.factors["noise"].to_elbo(
            mean=self.nodes["linear_predictor_adjacency"],
            x=self.nodes["noisy_linear_predictor_adjacency"],
            variance=self.parameters["noise_adjacency"].value()
        )
        elbo += self.factors["comparison_gaussian"].to_elbo(
            mean=self.nodes["linear_predictor_covariate"],
            x=self.nodes["covariates_continuous"],
            variance=self.parameters["noise_covariate"].value()
        )
        return elbo / (self.N ** 2)

    def elbo2(self):
        # prior contribution: cross-entropy + entropy of nodes
        prior_contribution = self.factors["latent_prior"].to_elbo(self.nodes["latent"])
        prior_contribution += self.factors["heterogeneity_prior"].to_elbo(self.nodes["heterogeneity"])
        # covariate contribution
        m = tf.tensordot(self.nodes["latent"].mean(), self.parameters["weight"].value(), 1) + \
            self.parameters["bias"].value()
        v = tf.tensordot(self.nodes["latent"].variance(), self.parameters["weight"].value() ** 2, 1)
        covariate_contribution = self.nodes["covariates_continuous"].mean() ** 2
        covariate_contribution += self.nodes["covariates_continuous"].variance_safe()
        covariate_contribution += m ** 2 + v
        covariate_contribution += -2. * self.nodes["covariates_continuous"].mean() * m
        covariate_contribution /= self.parameters["noise_covariate"].value()
        covariate_contribution += tf.math.log(self.parameters["noise_covariate"].value())
        covariate_contribution *= -0.5
        covariate_contribution = tf.reduce_sum(
            tf.where(self.nodes["covariates_continuous"].is_uniform(), 0., covariate_contribution)
        )
        covariate_contribution += GaussianArray.from_array(m, v).entropy()
        # adjacency contribution
        adjacency_contribution = self.factors["noise"].to_elbo(
            mean=self.nodes["linear_predictor_adjacency"],
            x=self.nodes["noisy_linear_predictor_adjacency"],
            variance=self.parameters["noise_adjacency"].value()
        )
        return (prior_contribution + covariate_contribution + adjacency_contribution) / (self.N ** 2)

    def pass_and_elbo(self):
        self.propagate(1)
        return self.elbo2()

    def propagate(self, n_iter=1):
        for _ in range(n_iter):
            self.forward_adjacency()
            self.backward_adjacency()
            self.forward_covariate()
            self.backward_covariate()

    def predict_covariates(self):
        # might want to add the variance?
        return self.factors["comparison_gaussian"].to_x(
            mean=self.nodes["linear_predictor_covariate"],
            variance=self.parameters["noise_covariate"].value()
        )

    def links_proba(self):
        return self.factors["adjacency"].to_result(self.nodes["noisy_linear_predictor_adjacency"])

    def _parameters(self):
        return {
            name: parameter._value
            for name, parameter in self.parameters.items()
        }