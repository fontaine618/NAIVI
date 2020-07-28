import tensorflow as tf


# TODO: Maybe refactor into a single class with different constructors (e.g. using a transformation argument)
# TODO: or abstract/child  classes


class ParameterArray:

    def __init__(self, array):
        self._value = tf.Variable(array)

    def value(self):
        return self._value

    def assign_add(self, v):
        self._value.assign_add(v)


class ParameterArrayLogScale:

    def __init__(self, array):
        self._value = tf.Variable(tf.math.log(array))

    def value(self):
        return tf.math.exp(self._value)

    def assign_add(self, v):
        self._value.assign_add(v)


