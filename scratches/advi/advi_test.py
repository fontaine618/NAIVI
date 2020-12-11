from NNVI.utils.gen_data import generate_dataset
from NNVI.utils.data import JointDataset
from NNVI.advi.model import ADVI
from NNVI.mle.model import MLE
from NNVI.vimc.model import VIMC

# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 500
K = 5
p_cts = 10
p_bin = 10
var_cts = 1.
missing_rate = 0.1
alpha_mean = -1.85

Z, a, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_rate=missing_rate,
    alpha_mean=alpha_mean, seed=1
)

train = JointDataset(i0, i1, A, X_cts, X_bin)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing)



# -----------------------------------------------------------------------------
# VIMC fit
# -----------------------------------------------------------------------------

self = VIMC(K, N, p_cts, p_bin)
self.fit(train, test, Z, batch_size=len(train), eps=1.e-6, max_iter=100, lr=0.01, n_sample=100)

# + tuning ...
# 24   | 0.5507     0.9817     0.0000    | 0.0877     0.4603    | 0.2208     1.0531     0.0000

# -----------------------------------------------------------------------------
# ADVI fit
# -----------------------------------------------------------------------------

self = ADVI(K, N, p_cts, p_bin)
self.fit(train, test, Z, batch_size=len(train), eps=1.e-6, max_iter=1000, lr=0.1)

# 250  | -0.5721    0.9759     0.0000    | 0.0491     0.2537    | -0.5593    1.0428     0.0000
# 233  | -0.2805    0.9764     0.7503    | 0.0759     0.3966    | -0.2809    1.1869     0.7247
# -----------------------------------------------------------------------------
# MLE fit
# -----------------------------------------------------------------------------

self = MLE(K, N, p_cts, p_bin)
self.fit(train, test, Z, batch_size=len(train), eps=1.e-6, max_iter=1000, lr=0.1)

# 172  | 0.5042     0.9623     0.0000    | 0.0606     0.2624    | 0.5143     1.0312     0.0000
# 251  | 0.2382     0.9665     0.7508    | 0.1277     0.4567    | 0.2433     1.1893     0.7246

