import numpy as np
import pandas as pd
import torch
import scipy.io
import matplotlib.pyplot as plt

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

# subsample
n = 500
N = X.shape[0]
i = np.random.randint(0, N, n)
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
which = n_per_tag > 15
X = X[:, which]

# do data
i = torch.tril_indices(n, n, -1)
i0 = i[0, :]
i1 = i[1, :]
A = A[i0, i1]
train = JointDataset(i0,
                     i1,
                     torch.Tensor(A).float(),
                     None,
                     torch.Tensor(X).float(),
                     return_missingness=False, cuda=True)


model = ADVI(K=10, p_cts=0, p_bin=X.shape[1], N=n)
model.fit(train=train)