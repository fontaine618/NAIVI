import tensorflow as tf


class ParameterArray:

    _n = 0

    def __init__(self, init, fixed=False, lr=0.01, name="unnamed"):
        ParameterArray._n += 1
        self._value = tf.Variable(init, name=str(ParameterArray._n) + "_" + name)
        self.grad = tf.zeros_like(init)
        self.fixed = fixed
        self.lr = lr

    def value(self):
        return self._value

    def assign_add(self, v):
        self._value.assign_add(v)

    def step(self):
        if not self.fixed:
            self.assign_add(self.grad * self.lr)


class ParameterArrayLogScale(ParameterArray):

    def __init__(self, init, fixed=False, lr=0.01, name=""):
        super().__init__(tf.math.log(init), fixed, lr, name)

    def value(self):
        return tf.math.exp(self._value)


