from missingpy import MissForest as MF
from ..mice.model import MICE


class MissForest(MICE):

    def __init__(self, K, N, p_cts, p_bin):
        super().__init__(K, N, p_cts, p_bin)
        self.model = MF()