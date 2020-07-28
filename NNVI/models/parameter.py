import tensorflow as tf


class ParameterArray:

    def __init__(self, init):
        self._value = tf.Variable(init)
        self.grad = tf.zeros_like(init)

    def value(self):
        return self._value

    def assign_add(self, v):
        self._value.assign_add(v)

    def step(self, lr):
        self.assign_add(self.grad * lr)


class ParameterArrayLogScale(ParameterArray):

    def __init__(self, init):
        super().__init__(tf.math.log(init))

    def value(self):
        return tf.math.exp(self._value)


