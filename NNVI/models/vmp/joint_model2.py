import tensorflow as tf

from models.vmp.vmp_factors2 import VMPFactor
from models.distributions.gaussianarray import GaussianArray
from models.distributions.bernoulliarray import BernoulliArray
from NNVI.models.vmp.vmp_factors2 import Prior, WeightedSum
from models.vmp.compound_factors import NoisyProbit, InnerProductModel, GLM


class JointModel2(VMPFactor):

    def __init__(self, K, A, X_cts, X_bin):
        super().__init__()
        self._deterministic = False
        self._check_input(K, A, X_cts, X_bin)

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
            variance=1.
        )
        self.heterogeneity_prior = Prior(
            child=self.heterogeneity,
            mean=0.,
            variance=1.
        )
        self.mean_model = WeightedSum(
            parent=self.positions,
            child=self.covariate_mean,
            bias=tf.zeros((1, self.p)),
            weight=tf.ones((self.K, self.p))
        )
        self.covariate_model = GLM(
            parent=self.covariate_mean,
            child_cts=self.covariate_continuous,
            child_bin=self.covariate_binary,
            variance_cts=tf.ones((1, self.p_cts)),
            variance_bin=tf.ones((1, self.p_bin))
        )
        self.inner_product_model = InnerProductModel(
            positions=self.positions,
            heterogeneity=self.heterogeneity,
            linear_predictor=self.adjacency_mean
        )
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

        self._initialize()
        self._initialized = False

    def _check_input(self, K, A, X_cts, X_bin):
        N = A.shape[0]
        p_cts = X_cts.shape[1]
        p_bin = X_bin.shape[1]
        self.N = N
        self.K = K
        self.p_cts = p_cts
        self.p_bin = p_bin
        self.p = p_cts + p_bin

    def _initialize(self):
        # send prior messages (need to be done only once)
        self.position_prior.init_child()
        self.heterogeneity_prior.init_child()

    def forward_adjacency(self):
        self.inner_product_model.forward(self._initialized)
        if not self._initialized:
            self._initialized = True
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
            self.forward_covariate()
            self.backward_covariate()

    def elbo(self):
        self.forward_covariate()
        self.forward_adjacency()
        elbo = 0.0
        for name, factor in self._factors.items():
            elbo += factor.to_elbo()
        return elbo / self.N ** 2

    def parameters_value(self):
        return {
            name: parm._value
            for name, parm in self.parameters().items()
            if not parm.fixed
        }

    def compute_gradient(self):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.parameters_value())
            target = self.elbo()
            for var in tape.watched_variables():
                print(var.name)
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