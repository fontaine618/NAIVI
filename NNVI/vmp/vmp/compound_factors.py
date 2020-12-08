import tensorflow as tf

from NNVI.vmp import BernoulliArray
from NNVI.vmp.distributions.gaussianarray import GaussianArray
from NNVI.vmp.vmp.vmp_factors2 import VMPFactor, AddVariance, Probit, Product, ExpandTranspose, Concatenate, Sum, Split, Logistic


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
        self._noise.forward()
        # update child (just to update the prediction; does nothing to vmp)
        self._probit.forward()

    def backward(self):
        # update noisy
        self._probit.backward()
        # compute parent
        self._noise.backward()

    def to_elbo(self):
        return self._noise.to_elbo()

    def predict(self):
        self.forward()
        return self._probit.message_to_child


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

    def forward(self):
        self._product.forward()
        self._expand_transpose.forward()
        self._concatenate.forward()
        self._sum.forward()

    def backward(self):
        self._sum.backward()
        self._concatenate.backward()
        self._expand_transpose.backward()
        self._product.backward()


class GLM(VMPFactor):

    def __init__(
            self,
            parent: GaussianArray,
            child_cts: GaussianArray,
            child_bin: BernoulliArray,
            variance_cts=None,
            variance_bin=None,
            bin_model="NoisyProbit"
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
        self._split = Split(
            parts={"cts": self._lin_pred_cts, "bin": self._lin_pred_bin},
            vector=self.parent
        )
        self._model_cts = AddVariance(
            parent=self._lin_pred_cts,
            child=self.child_cts,
            variance=variance_cts
        )
        if bin_model == "Logistic":
            self._model_bin = Logistic(
                parent=self._lin_pred_bin,
                child=self.child_bin,
            )
        else:
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
        self._split.forward()
        self._model_cts.forward()
        self._model_bin.forward()

    def backward(self):
        self._model_bin.backward()
        self._model_cts.backward()
        self._split.backward()

    def to_elbo(self, **kwargs):
        return self._model_bin.to_elbo() + self._model_cts.to_elbo()

    def predict(self):
        self._split.forward()
        return {
            "continuous": self._model_cts.predict(),
            "binary": self._model_bin.predict()
        }
