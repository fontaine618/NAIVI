import numpy as np
import torch
import time
from facebook.data import get_data
from NAIVI.utils.data import JointDataset
from NAIVI.advi.model import ADVI
from NAIVI.mle.model import MLE
from NAIVI.vimc.model import VIMC
from NAIVI.mice.model import MICE

DATA_PATH = "./facebook/data/raw/"

center = 0
missing_rate = [0.4, 0.8]
seed = 0
K = 10
alpha_mean = -1.85
sparsity = 0.50
K_model = 10
algo = "MLE"
max_iter = 100
n_sample = 1
eps = 1.0e-6
lr = 0.05
reg = 10 ** np.linspace(1.0, -1.0, 10)


# get data
i0, i1, A, X_cts, X_bin = get_data(DATA_PATH, center)
# recover data
N = X_bin.size(0)
p_cts = 0
p_bin = X_bin.size(1)
p = p_cts + p_bin
density = A.mean().item()
# insert missing values
torch.manual_seed(seed)
p_non_zero = int(p * sparsity)
prob1 = torch.ones((1, p)) * (missing_rate[1] - missing_rate[0])
prob1[0, p_non_zero:] = 0.0
prob = torch.ones_like(X_bin) * missing_rate[0]
prob = prob + X_bin * prob1
mask = torch.rand_like(X_bin) < prob
X_bin_missing = torch.where(mask, np.nan, X_bin)
X_bin = torch.where(~mask, np.nan, X_bin)
X_cts_missing = None
# print
print("=" * 80)
print(center, seed, N, p_cts, p_bin, K, missing_rate, alpha_mean, K_model, algo)
input = (
    center,
    seed,
    N,
    p_cts,
    p_bin,
    K,
    *missing_rate,
    alpha_mean,
    sparsity,
    K_model,
    algo,
)
# dataset format
mnar = False if algo == "MICE" else True
train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar, test=True)
# initialization
B0 = torch.zeros((1, p))
B = torch.randn((K_model, p))
initial = {
    "bias": torch.cat([B0, B0], 1).cuda(),
    "weight": torch.cat([B, B*0.01], 1).cuda(),
    "positions": torch.randn((N, K_model)).cuda(),
    "heterogeneity": torch.randn((N, 1)).cuda() * 0.5 - 1.85,
}
# ---------------------------------------------------------------------
# initialize model
fit_args = {
    "eps": eps,
    "max_iter": max_iter,
    "lr": lr,
    "init": initial,
    "reg": reg,
    "batch_size": len(train),
}
if algo == "ADVI":
    model = ADVI(K_model, N, p_cts, p_bin, mnar=True)
elif algo == "VIMC":
    model = VIMC(K_model, N, p_cts, p_bin, mnar=True)
    fit_args["n_sample"] = n_sample
elif algo == "VMP":
    raise RuntimeError("VMP not implemented yet")
elif algo == "MICE":
    model = MICE(K_model, N, p_cts, p_bin)
    fit_args = {}
else:
    model = MLE(K_model, N, p_cts, p_bin, mnar=True)
# ---------------------------------------------------------------------
# fit model
t0 = time.time()
output = model.fit_path(train, test, **fit_args), density, time.time() - t0





train = JointDataset(i0, i1, A, X_cts, X_bin)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing)
initial = {
    "bias": B0.cuda(),
    "weight": B.cuda(),
    "positions": torch.randn((N, K_model)).cuda(),
    "heterogeneity": torch.randn((N, 1)).cuda() * 0.5 - 1.85,
}
fit_args["reg"] = 0.0
fit_args.pop("init")
model = ADVI(K_model, N, p_cts, p_bin, mnar=False)
model.init(**initial)
output = model.fit(train, test, **fit_args), density, time.time() - t0

output

model = MICE(K_model, N, p_cts, p_bin)
fit_args = {}
model.fit(train, test)


for n, p in model.model.named_parameters():
    print(n, p)
