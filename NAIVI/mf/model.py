from missingpy import MissForest as MF
from NAIVI import MICE


class MissForest(MICE):

    def __init__(self, K, N, p_cts, p_bin):
        super().__init__(K, N, p_cts, p_bin)
        self.model = MF()