import torch
import numpy as np
from NAIVI.utils.gen_data import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI.advi.model import ADVI
from NAIVI.mle.model import MLE
from NAIVI.vimc.model import VIMC
from NAIVI.mice.model import MICE
from NAIVI.mf import MissForest
from NAIVI.constant import Mean
from NAIVI.smoothing import NetworkSmoothing


# -----------------------------------------------------------------------------
# Create Data
# -----------------------------------------------------------------------------

N = 500
K = 5
p_cts = 100
p_bin = 0
var_cts = 1.
missing_rate = 0.10
alpha_mean = -1.85
seed = 0

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_rate=missing_rate,
    alpha_mean=alpha_mean, seed=seed
)

train = JointDataset(i0, i1, A, X_cts, X_bin)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing)
# -----------------------------------------------------------------------------
# MissForest fit
# -----------------------------------------------------------------------------
self = NetworkSmoothing(K, N, p_cts, p_bin)
out = self.fit(train, test)
self = MissForest(K, N, p_cts, p_bin)
out = self.fit(train, test)
self = MICE(K, N, p_cts, p_bin)
out = self.fit(train, test)
self = Mean(K, N, p_cts, p_bin)
out = self.fit(train, test)



#
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

self = VIMC(K, N, p_cts, p_bin, n_samples=1)
self.init(positions=Z.cuda(), heterogeneity=alpha.cuda())
self.fit(train, test, Z.cuda(), batch_size=len(train), eps=1.e-6,
         max_iter=200, lr=0.01)

# -----------------------------------------------------------------------------
# ADVI fit
# -----------------------------------------------------------------------------

self = ADVI(K, N, p_cts, p_bin)
self.init(positions=Z.cuda(), heterogeneity=alpha.cuda())
self.fit(train, test, Z.cuda(), batch_size=len(train), eps=1.e-6, max_iter=200, lr=0.01)

# -----------------------------------------------------------------------------
# MLE fit
# -----------------------------------------------------------------------------

self = MLE(K, N, p_cts, p_bin)
self.init(positions=Z.cuda(), heterogeneity=alpha.cuda())
self.fit(train, test, Z, batch_size=len(train), eps=1.e-6, max_iter=200, lr=0.1)


# -----------------------------------------------------------------------------
# MICE fit
# -----------------------------------------------------------------------------
missing_rate = 0.28641
var_cts = 1.
seed = 0

Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0 = generate_dataset(
    N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cts, missing_rate=missing_rate,
    alpha_mean=alpha_mean, seed=seed
)

train = JointDataset(i0, i1, A, X_cts, X_bin)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing)
self = MICE(K, N, p_cts, p_bin)
out = self.fit(train, test)

from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import BayesianRidge
from sklearn.impute import IterativeImputer
self = MICE(K, N, p_cts, p_bin)
estimator = BayesianRidge()
self.model = IterativeImputer(
    random_state=1, estimator=estimator, imputation_order="ascending", max_iter=100
)
out = self.fit(train, test)
print(missing_rate, out[-2])





from statsmodels.imputation import mice
import pandas as pd
data = pd.DataFrame(train[:][4].cpu().numpy())
data.columns = [f"X{col}" for col in data.columns]
imp = mice.MICEData(data)
# add all imputations
for col in data.columns:
    imp.set_imputer(col, " + ".join([c for c in data.columns if c is not col]))
imp.update_all()
imp.data