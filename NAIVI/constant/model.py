from sklearn.impute import SimpleImputer
from ..mice.model import MICE


class Mean(MICE):

    def __init__(self, K, N, p_cts, p_bin):
        super().__init__(K, N, p_cts, p_bin)
        self.model = SimpleImputer(strategy="mean")
