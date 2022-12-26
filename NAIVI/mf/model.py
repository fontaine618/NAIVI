# patching version incompatibility between missingpy and sklearn ...
import sys
import sklearn.neighbors._base

# need to add this function to sklearn.neighbors._base
# since it got removed in sklearn 0.22.1
def _check_weights(weights):
    """Check to make sure weights are valid"""
    if weights in (None, 'uniform', 'distance'):
        return weights
    elif callable(weights):
        return weights
    else:
        raise ValueError("weights not recognized: should be 'uniform', "
                         "'distance', or a callable function")

sklearn.neighbors._base._check_weights = _check_weights
sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base

from missingpy import MissForest as MF
from NAIVI import MICE


class MissForest(MICE):

    def __init__(self, K, N, p_cts, p_bin):
        super().__init__(K, N, p_cts, p_bin)
        self.model = MF()