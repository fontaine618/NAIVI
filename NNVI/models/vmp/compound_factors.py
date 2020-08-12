import tensorflow as tf

from models import BernoulliArray
from models.distributions.gaussianarray import GaussianArray
from models.vmp.vmp_factors2 import VMPFactor, AddVariance, Probit, Product, ExpandTranspose, Concatenate, Sum


class NoisyProbit(VMPFactor):

    def __init__(self, child: BernoulliArray, parent: GaussianArray, variance=tf.ones((1, 1))):
        super().__init__()
        self._deterministic = False
        self.shape = child.shape()
        # nodes
        self.child = child
        self.parent = parent
        # hidden nodes
        self._noisy = GaussianArray.uniform(self.shape)
        self._nodes.update({"noisy": self._noisy})
        # factors
        self._noise = AddVariance(
            parent=parent,
            child=self._noisy,
            variance=variance
        )
        self._probit = Probit(
            parent=self._noisy,
            child=child
        )
        self._factors.update({
            "noise": self._noise,
            "probit": self._probit
        })

    def forward(self):
        # update noisy
        self._noise.to_child()
        # update child (just to update the prediction; does nothing to vmp)
        self._probit.to_child()

    def backward(self):
        # update noisy
        self._probit.to_parent()
        # compute parent
        self._noise.to_parent()

    def to_elbo(self):
        return self._noise.to_elbo()


class InnerProductModel(VMPFactor):

    def __init__(self, positions, heterogeneity, linear_predictor):
        super().__init__()
        self._deterministic = True
        N, K = positions.shape()
        # neighboring nodes
        self.positions = positions
        self.heterogeneity = heterogeneity
        self.linear_predictor = linear_predictor
        # hidden nodes
        self._products = GaussianArray.uniform((N, N, K))
        self._heterogeneity_expanded = GaussianArray.uniform((N, N, 2))
        self._vector = GaussianArray.uniform((N, N, K + 2))
        self._nodes.update({
            "products": self._products,
            "heterogeneity_expanded": self._heterogeneity_expanded,
            "vector": self._vector
        })
        # hidden factors
        self._product = Product(
            parent=self.positions,
            child=self._products
        )
        self._expand_transpose = ExpandTranspose(
            child=self._heterogeneity_expanded,
            parent=self.heterogeneity
        )
        self._concatenate = Concatenate(
            parts={
                "products": self._products,
                "heterogeneity_expanded": self._heterogeneity_expanded
            },
            vector=self._vector
        )
        self._sum = Sum(
            parent=self._vector,
            child=self.linear_predictor
        )
        self._factors.update({
            "product": self._product,
            "expand_transpose": self._expand_transpose,
            "concatenate": self._concatenate,
            "sum": self._sum
        })

    def forward(self, initialized):
        if initialized:
            self._product.to_child()
        else:
            init_message = GaussianArray.from_array(
                mean=tf.random.normal(self._product.child.shape(), 0., 1.),
                variance=1.
            )
            self._product.child.update(self._product.message_to_child, init_message)
            self._product.message_to_child = init_message
        self._expand_transpose.to_child()
        self._concatenate.to_vector()
        self._sum.to_child()

    def backward(self):
        self._sum.to_parent()
        self._concatenate.to_parts()
        self._expand_transpose.to_parent()
        self._product.to_parent()


class GLM(VMPFactor):

    def __init__(
            self,
            parent: GaussianArray,
            child_cts: GaussianArray,
            child_bin: BernoulliArray,
            variance_cts=None,
            variance_bin=None
    ):
        super().__init__()
        self._deterministic = False
        # nodes
        self.parent = parent
        self.child_cts = child_cts
        self.child_bin = child_bin
        # dimensions
        N, K = parent.shape()
        p_cts, p_bin = child_cts.shape()[1], child_bin.shape()[1]
        if variance_cts is None:
            variance_cts = tf.ones((1, p_cts))
        if variance_bin is None:
            variance_bin = tf.ones((1, p_bin))
        # hidden nodes
        self._lin_pred_cts = GaussianArray.uniform((N, p_cts))
        self._lin_pred_bin = GaussianArray.uniform((N, p_bin))
        self._nodes.update({
            "lin_pred_cts": self._lin_pred_cts,
            "lin_pred_bin": self._lin_pred_bin
        })
        # factors
        self._split = Concatenate(
            parts={"cts": self._lin_pred_cts, "bin": self._lin_pred_bin},
            vector=self.parent
        )
        self._model_cts = AddVariance(
            parent=self._lin_pred_cts,
            child=self.child_cts,
            variance=variance_cts
        )
        self._model_bin = NoisyProbit(
            parent=self._lin_pred_bin,
            child=self.child_bin,
            variance=variance_bin
        )
        self._factors.update({
            "split": self._split,
            "model_cts": self._model_cts,
            "model_bin": self._model_bin
        })

    def forward(self):
        self._split.to_parts()
        self._model_cts.to_child()
        self._model_bin.forward()

    def backward(self):
        self._model_bin.backward()
        self._model_cts.to_parent()
        self._split.to_vector()

    def to_elbo(self, **kwargs):
        return self._model_bin.to_elbo() + self._model_cts.to_elbo()