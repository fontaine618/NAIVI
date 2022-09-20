import numpy as np
import pandas as pd
import torch
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns

from NAIVI.utils.data import JointDataset
from NAIVI import ADVI, MLE, MAP, VIMC, MCMC, GLM, MissForest, MICE, NetworkSmoothing, Mean
plt.style.use("seaborn-whitegrid")

FILE_PATH = "/home/simon/Documents/NAIVI/NAIVI_blogcatalog/data/BlogCatalog.mat"

mat_obj = scipy.io.loadmat(FILE_PATH)
network = mat_obj["Network"]
labels = mat_obj["Label"]
attributes = mat_obj["Attributes"]
classes = mat_obj["Class"]

X = pd.DataFrame(attributes.todense())
X = X.gt(0.).replace({True:1, False:0})

A = pd.DataFrame(network.todense())

del mat_obj, network, labels, attributes, classes

# subsample
N = 500
n = X.shape[0]

n_friends = A.sum(1)
i = n_friends.argsort()[::-1][:N]
X = X.values[i, :]
A = A.values[i, :][:, i]

n_friends = A.sum(1)
n_per_user = X.sum(1)
n_per_tag = X.sum(0)

plt.cla()
pd.Series(n_per_user).hist(bins=np.arange(0, 200, 2))
plt.show()

plt.cla()
pd.Series(n_per_tag).hist(bins=np.arange(0, 50, 2))
plt.show()

plt.cla()
pd.Series(n_friends).hist(bins=np.arange(0, 50, 2))
plt.show()

# subset X columns
which = n_per_tag > 100
which.sum()
X = X[:, which]

# do data
i = torch.tril_indices(N, N, -1)
i0 = i[0, :]
i1 = i[1, :]
A = A[i0, i1]

# to torch
X = torch.Tensor(X).double()
A = torch.Tensor(A).double().reshape((-1, 1))

# insert missing values
missing_rate0 = 0.5
missing_rate1 = 0.2
missing_rate_mcar = 0.4
missing_rate = X * missing_rate1 + (1-X) * missing_rate0
missing_rate[:, range(10)] = missing_rate_mcar
missing = torch.rand_like(X) < missing_rate

X_obs = torch.where(missing, np.nan, X)
X_mis = torch.where(missing, X, np.nan)


train = JointDataset(i0, i1, A, None, X_obs, return_missingness=True, cuda=True)
test = JointDataset(i0, i1, A, None, X_mis, return_missingness=True, cuda=True)
model = ADVI(K=5, p_cts=0, p_bin=X.shape[1], N=N, mnar=True, network_weight=0.)
out = model.fit(train=train, test=test, eps=0.0001, lr=0.01, reg=100.,
                optimizer="Rprop", max_iter=200)
for k, v in out.items():
    print(f"{k[0]:<10} {k[1]:<16} {v:4f}")




# train = JointDataset(i0, i1, A, None, X_obs, return_missingness=False, cuda=False)
# test = JointDataset(i0, i1, A, None, X_mis, return_missingness=False, cuda=False)
# model = ADVI(K=5, p_cts=0, p_bin=X.shape[1], N=N, mnar=False)
# out = model.fit(train=train, test=test, eps=0.0001, lr=0.01,
#                 optimizer="Rprop", max_iter=200)
# for k, v in out.items():
#     print(f"{k[0]:<10} {k[1]:<16} {v:4f}")




BC = model.covariate_weight.data
BCBCt = BC @ BC.T
BCnorm = BC.pow(2).sum(1).sqrt()
BCnorm = torch.where(BCnorm == 0., torch.ones_like(BCnorm), BCnorm)
BCBCt = BCBCt / (BCnorm.reshape((-1, 1)) * BCnorm.reshape((1, -1)))

plt.imshow(BCBCt.detach().cpu().numpy(), cmap="RdBu", vmin=-1, vmax=1)
plt.colorbar()
plt.grid(None)
plt.axhline(23.5, color="white")
plt.axvline(23.5, color="white")
plt.axhline(23.5+10, color="white")
plt.axvline(23.5+10, color="white")
plt.tight_layout()
plt.show()