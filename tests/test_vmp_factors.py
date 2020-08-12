from unittest import TestCase
import tensorflow as tf

from models.distributions.gaussianarray import GaussianArray
from NNVI.models.vmp.vmp_factors import Sum


class TestPrior(TestCase):
    pass


class TestSum(TestCase):

    def test_to_sum(self):
        factor = Sum((5, 3), (5, ))
        x = GaussianArray.from_array(tf.random.normal((5, 3), 0., 1.), tf.ones((5, 3)))
        factor.to_sum(x)
        tf.debugging.assert_equal(factor.message_to_sum.mean(), tf.math.reduce_sum(x.mean(), -1))

    def test_to_x(self):
        self.fail()


class TestProduct(TestCase):
    def test_to_product(self):
        self.fail()

    def test_to_a(self):
        self.fail()

    def test_to_b(self):
        self.fail()
