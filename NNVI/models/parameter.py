import tensorflow as tf
import numpy as np


class ParameterArray:

    def __init__(self, array):
        self.value = array

    def clear_incoming_messages(self):
        pass

    def gradient_step(self, step):
        pass
