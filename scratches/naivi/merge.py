from NNVI.utils.gen_data import generate_dataset
from NNVI.utils.data import JointDataset
from NNVI.advi.model import ADVI
from NNVI.mle.model import MLE
from NNVI.vimc.model import VIMC
from NNVI.mice.model import MICE


# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 50
K = 5
p_cts = 100
p_bin = 0
var_cts = 1.
missing_rate = 0.10
alpha_mean = -3.00
seed = 0

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_rate=missing_rate,
    alpha_mean=alpha_mean, seed=seed
)

train = JointDataset(i0, i1, A, X_cts, X_bin)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing)

# import pandas as pd
# X = pd.DataFrame(X_cts_missing.numpy())
# var = X.var()
# print(var.min(), var.max(), var.mean())
#
# corr = X.corr()
#
# import matplotlib.pyplot as plt
# plt.imshow(corr, vmin=-1., vmax=1., cmap="RdBu")
# plt.colorbar()
# plt.savefig("asdas.pdf")
# plt.clf()
# corr[corr.abs() ==1.] = 0.
# corr.abs().max().max()
# -----------------------------------------------------------------------------
# VIMC fit
# -----------------------------------------------------------------------------

self = VIMC(K, N, p_cts, p_bin, 10)
self.init(positions=Z, heterogeneity=alpha)
self.fit(train, test, Z, batch_size=len(train), eps=1.e-6,
         max_iter=200, lr=0.01)

# -----------------------------------------------------------------------------
# ADVI fit
# -----------------------------------------------------------------------------

self = ADVI(K, N, p_cts, p_bin)
self.init(positions=Z, heterogeneity=alpha)
self.fit(train, test, Z, batch_size=len(train), eps=1.e-6, max_iter=200, lr=0.01)

# -----------------------------------------------------------------------------
# MLE fit
# -----------------------------------------------------------------------------

self = MLE(K, N, p_cts, p_bin)
self.init(positions=Z, heterogeneity=alpha)
self.fit(train, test, Z, batch_size=len(train), eps=1.e-6, max_iter=200, lr=0.1)

# -----------------------------------------------------------------------------
# MICE fit
# -----------------------------------------------------------------------------

self = MICE(K, N, p_cts, p_bin)
self.fit(train, test)



