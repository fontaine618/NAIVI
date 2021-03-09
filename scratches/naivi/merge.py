from NAIVI.utils.gen_data import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI
from NAIVI import MLE
from NAIVI import VIMC
from NAIVI import MICE
import torch


# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 500
K = 5
p_cts = 100
p_bin = 0
var_cts = 1.
missing_rate = 0.20
alpha_mean = -1.85
seed = 0

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_rate=missing_rate,
    alpha_mean=alpha_mean, seed=seed
)

mnar = False
train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing,
                    return_missingness=mnar, test=True)
if mnar:
    B0 = torch.cat([B0, B0], 1)
    B = torch.cat([B, B], 1)

init = {"positions": Z, "heterogeneity": alpha, "bias": B0, "weight": B}
init = {k: v * (0.8 + 0.2*torch.rand_like(v)) for k, v in init.items()}
# -----------------------------------------------------------------------------
# VIMC fit
# -----------------------------------------------------------------------------

self = VIMC(K, N, p_cts, p_bin, n_samples=1, mnar=mnar)
self.init(**init)
self.fit(train, test, Z, batch_size=len(train),
         eps=5.e-6, max_iter=2000, lr=1.)

# -----------------------------------------------------------------------------
# ADVI fit
# -----------------------------------------------------------------------------

self = ADVI(K, N, p_cts, p_bin, mnar=mnar)
self.init(**init)
self.fit(train, test, Z, batch_size=len(train),
         eps=5.e-6, max_iter=2000, lr=1.)

# -----------------------------------------------------------------------------
# MLE fit
# -----------------------------------------------------------------------------

self = MLE(K, N, p_cts, p_bin, mnar=mnar)
self.init(**init)
self.fit(train, test, Z, batch_size=len(train),
         eps=5.e-6, max_iter=2000, lr=0.1)

# -----------------------------------------------------------------------------
# MICE fit
# -----------------------------------------------------------------------------

self = MICE(K, N, p_cts, p_bin)
self.fit(train, test)



