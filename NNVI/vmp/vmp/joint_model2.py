import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error

from NNVI.vmp.vmp.vmp_factors2 import VMPFactor
from NNVI.vmp.distributions.gaussianarray import GaussianArray
from NNVI.vmp.distributions.bernoulliarray import BernoulliArray
from NNVI.vmp.vmp.vmp_factors2 import Prior, WeightedSum, Logistic
from NNVI.vmp.vmp.compound_factors import NoisyProbit, InnerProductModel, GLM
from NNVI.vmp.utils import invariant_matrix_distance, projection_distance


class JointModel2(VMPFactor):

    def __init__(
            self,
            K, A,
            X_cts=None, X_bin=None,
            link_model="NoisyProbit", bin_model="NoisyProbit",
            initial={
                "bias": None,
                "weights": None,
                "positions": None,
                "heterogeneity": None
            }
    ):
        super().__init__()
        self._deterministic = False
        self._check_input(K, A, X_cts, X_bin)

        # initial values
        if initial["bias"] is None:
            initial["bias"] = tf.zeros((1, self.p))
        if initial["weights"] is None:
            initial["weights"] = tf.ones((self.K, self.p))
        if initial["positions"] is None:
            initial["positions"] = tf.random.normal((self.N, self.K), 0., 1.)
        if initial["heterogeneity"] is None:
            initial["heterogeneity"] = tf.random.normal((self.N, 1), 0., 1.)

        # prepare nodes
        self.positions = GaussianArray.uniform((self.N, self.K))
        self.heterogeneity = GaussianArray.uniform((self.N, 1))
        self.covariate_mean = GaussianArray.uniform((self.N, self.p))
        self.adjacency_mean = GaussianArray.uniform((self.N, self.N))
        self.covariate_continuous = GaussianArray.observed(X_cts)
        self.covariate_binary = BernoulliArray.observed(X_bin)
        self.links = BernoulliArray.observed(A)
        self._nodes = {
            "positions": self.positions,
            "heterogeneity": self.heterogeneity,
            "covariate_mean": self.covariate_mean,
            "adjacency_mean": self.adjacency_mean,
            "covariate_continuous": self.covariate_continuous,
            "covariate_binary": self.covariate_binary,
            "links": self.links
        }

        # prepare factors
        self.position_prior = Prior(
            child=self.positions,
            mean=0.,
            variance=1.,
            initial=initial["positions"],
            name="position_prior"
        )
        self.heterogeneity_prior = Prior(
            child=self.heterogeneity,
            mean=-2.,
            variance=1.,
            initial=initial["heterogeneity"],
            name="heterogeneity_prior"
        )
        self.inner_product_model = InnerProductModel(
            positions=self.positions,
            heterogeneity=self.heterogeneity,
            linear_predictor=self.adjacency_mean
        )
        if self.p > 0:
            self.mean_model = WeightedSum(
                parent=self.positions,
                child=self.covariate_mean,
                bias=initial["bias"],
                weight=initial["weights"]
            )
            self.covariate_model = GLM(
                parent=self.covariate_mean,
                child_cts=self.covariate_continuous,
                child_bin=self.covariate_binary,
                variance_cts=tf.ones((1, self.p_cts)),
                variance_bin=tf.ones((1, self.p_bin)),
                bin_model=bin_model
            )
        else:
            self.mean_model = VMPFactor()
            self.covariate_model = VMPFactor()
        if link_model == "Logistic":
            self.adjacency_model = Logistic(
                parent=self.adjacency_mean,
                child=self.links
            )
        else:
            self.adjacency_model = NoisyProbit(
                parent=self.adjacency_mean,
                child=self.links,
                variance=1.
            )
        self._factors = {
            "position_prior": self.position_prior,
            "heterogeneity_prior": self.heterogeneity_prior,
            "mean_model": self.mean_model,
            "covariate_model": self.covariate_model,
            "inner_product_model": self.inner_product_model,
            "adjacency_model": self.adjacency_model
        }

        self.elbo = -np.inf
        # self._initialize()
        # self._initialized = False

    def _check_input(self, K, A, X_cts=None, X_bin=None):
        N = A.shape[0]
        p_cts = 0
        p_bin = 0
        if X_cts is not None:
            p_cts = X_cts.shape[1]
        if X_bin is not None:
            p_bin = X_bin.shape[1]
        self.N = N
        self.K = K
        self.p_cts = p_cts
        self.p_bin = p_bin
        self.p = p_cts + p_bin

    def _initialize(self):
        self.position_prior.forward()
        self.heterogeneity_prior.forward()

    def set_lr(self, lr):
        for p in self.parameters().values():
            p.lr = lr

    def forward_adjacency(self):
        self.inner_product_model.forward()
        self.adjacency_model.forward()

    def forward_covariate(self):
        self.mean_model.to_child()
        self.covariate_model.forward()

    def backward_adjacency(self):
        self.adjacency_model.backward()
        self.inner_product_model.backward()

    def backward_covariate(self):
        self.covariate_model.backward()
        self.mean_model.to_parent()

    def propagate(self, n_iter=1):
        for _ in range(n_iter):
            self.forward_adjacency()
            self.backward_adjacency()
            self.position_prior.forward()
            self.heterogeneity_prior.forward()
            if self.p > 0:
                self.forward_covariate()
                self.backward_covariate()

    def to_elbo(self):
        if self.p > 0:
            self.forward_covariate()
        self.forward_adjacency()
        elbo = 0.0
        for name, factor in self._factors.items():
            elbo += factor.to_elbo()
        self.elbo = elbo / (self.N * (self.N + self.p))
        return self.elbo

    def parameters_value(self):
        return {
            name: parm._value
            for name, parm in self.parameters().items()
            if not parm.fixed
        }

    def compute_gradient(self):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.parameters_value())
            target = self.to_elbo()
        grad = tape.gradient(target, self.parameters_value())
        for name, g in grad.items():
            self._parameters[name].grad = g

    def gradient(self):
        return {
            name: parm.grad
            for name, parm in self.parameters().items()
            if not parm.fixed
        }

    def step(self):
        for name, parm in self.parameters().items():
            parm.step()

    def fit(self, n_iter=100, n_vmp=10, n_gd=10, verbose=False,
            X_bin_missing=None, X_cts_missing=None, positions_true=None):
        if verbose:
            print("{:10}  {:10}  {:10}  {:10}  {:10}  {:10}  {:10}  {:10}  {:10}".format(
                "Iteration",
                "VMP Iter.",
                "VMP ELBO",
                "GD Iter.",
                "GD ELBO",
                "MSE",
                "AUROC",
                "Dist",
                "Dist(proj)"
            ))
        elbo = self.to_elbo()
        #self.propagate(100)
        self.to_elbo()
        # Evaluate
        err = self.covariate_metrics(X_cts_missing, X_bin_missing)
        dist = self.latent_distance(positions_true)
        if verbose:
            print("{:10}  {:10}  {:10.6f}  {:10}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}".format(
                0,
                0,
                elbo.numpy(),
                0,
                self.elbo.numpy(),
                err["mse"],
                err["auroc"],
                dist["inv"],
                dist["proj"]
            ))
        elbo_pre_e_step = -np.inf
        elbo_post_m_step = -np.inf
        for i in range(n_iter):
            # M Step
            nm = 0
            for nm in range(1):
                self.propagate(n_vmp)
                self.to_elbo()
            if np.isnan(self.elbo):
                return
            elbo_pre_e_step = self.elbo
            # Evaluate
            err = self.covariate_metrics(X_cts_missing, X_bin_missing)
            dist = self.latent_distance(positions_true)
            # E Step
            ne = -1
            for ne in range(n_gd):
                self.compute_gradient()
                self.step()
            elbo_post_m_step = self.to_elbo()
            if verbose:
                print("{:10}  {:10}  {:10.6f}  {:10}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}".format(
                    i + 1,
                    n_vmp,
                    elbo_pre_e_step.numpy(),
                    ne + 1,
                    elbo_post_m_step.numpy(),
                    err["mse"],
                    err["auroc"],
                    dist["inv"],
                    dist["proj"]
                ))

    def predict_covariate(self):
        self.forward_covariate()
        return self.covariate_model.predict()

    def covariate_metrics(self, X_cts_missing=None, X_bin_missing=None):
        if self.p > 0:
            predictions = self.predict_covariate()
            if X_cts_missing is not None and tf.reduce_sum(tf.where(tf.math.is_nan(X_cts_missing), 0., 1.)) > 0.:
                mask = tf.where(tf.math.is_nan(X_cts_missing), False, True).numpy()
                mse = mean_squared_error(X_cts_missing.numpy()[mask], predictions["continuous"].numpy()[mask])
            else:
                mse = 0.
            if X_bin_missing is not None and tf.reduce_sum(tf.where(tf.math.is_nan(X_bin_missing), 0., 1.)) > 0.:
                mask = tf.where(tf.math.is_nan(X_bin_missing), False, True).numpy()
                auroc = roc_auc_score(X_bin_missing.numpy()[mask], predictions["binary"].numpy()[mask])
            else:
                auroc = 0.
        else:
            mse = 0.
            auroc = 0.
        return {
            "mse": mse,
            "auroc": auroc
        }

    def latent_distance(self, Z=None):
        if Z is None:
            return {
            "inv": 0.,
            "proj": 0.
            }
        return {
            "inv": invariant_matrix_distance(Z, self.positions.mean()).numpy(),
            "proj": projection_distance(Z, self.positions.mean()).numpy()
        }