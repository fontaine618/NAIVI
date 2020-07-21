class Distribution:

    def __init__(self):
        pass

    def is_point_mass(self):
        pass

    def is_uniform(self):
        pass

    def mean(self):
        pass

    def variance(self):
        pass

    def precision(self):
        pass

    def mean_times_precision(self):
        pass

    def mode(self):
        pass

    def natural(self):
        pass

    def mean_and_variance(self):
        pass

    def shape(self):
        pass

    def entropy(self):
        pass

    def negative_entropy(self):
        return - self.entropy()

    @classmethod
    def uniform(cls, *args, **kwargs):
        pass

    @classmethod
    def observed(cls, *args, **kwargs):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        return str(self)