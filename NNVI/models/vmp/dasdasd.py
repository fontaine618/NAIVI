import tensorflow as tf

from models import BernoulliArray
from models.distributions.gaussianarray import GaussianArray
from models.vmp.logistic_utils import sigmoid_integrals
from models.vmp.vmp_factors2 import VMPFactor


class Logistic(VMPFactor):

    def __init__(self, child: BernoulliArray, parent: GaussianArray):
        super().__init__()
        self._deterministic = True
        self.shape = child.shape()
        self.child = child
        self.parent = parent
        self.message_to_child = BernoulliArray.uniform(self.shape)
        self.message_to_parent = GaussianArray.uniform(self.shape)

    def to_parent(self):
        # Remove previous message ?
        x = self.parent / self.message_to_parent
        m, v = x.mean_and_variance()
        #m, v = self.parent.mean_and_variance()
        integrals = sigmoid_integrals(m, v, [0, 1])
        sd = tf.math.sqrt(v)
        exp1 = m * integrals[0] + sd * integrals[1]
        p = (exp1 - m * integrals[0]) / v
        mtp = m * p + self.child.proba() - integrals[0]

        message_to_parent = GaussianArray(p, mtp)
        self.parent.update(self.message_to_parent, message_to_parent)
        self.message_to_parent = message_to_parent

    def to_child(self):
        x = self.parent / self.message_to_parent
        m, v = x.mean_and_variance()
        integral = sigmoid_integrals(m, v, [0])[0]
        proba = tf.where(self.child.proba() == 1., integral, 1. - integral)
        self.message_to_child = BernoulliArray.from_array(proba)

    def to_elbo(self, **kwargs):
        pass