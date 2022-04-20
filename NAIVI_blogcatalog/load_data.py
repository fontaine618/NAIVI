import numpy as np
import pandas as pd
import torch
import scipy.io
import matplotlib.pyplot as plt
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
n_friends = A.sum(1)

n_per_user = X.sum(1)
n_per_tag = X.sum(0)

plt.cla()
n_per_user.hist(bins=np.arange(0, 200, 2))
plt.show()


plt.cla()
n_per_tag.hist(bins=np.arange(0, 200, 2))
plt.show()

plt.cla()
n_friends.hist(bins=np.arange(0, 200, 2))
plt.show()