import numpy as np
import time
import torch
from pypet import Parameter
from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI import ADVI, MLE, MAP, VIMC, MCMC, GLM, MissForest, MICE, NetworkSmoothing, Mean
from NAIVI.initialization import initialize
import os
import arviz
import pandas as pd


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def compute_true_values(B, B0, Z, alpha, i0, i1):
    # prepare true values for comparison
    ZZt_true = Z @ Z.T
    A_logit = alpha[i0] + alpha[i1] + torch.sum(Z[i0, :] * Z[i1, :], 1, keepdim=True)
    proba_true = torch.sigmoid(A_logit)
    Theta_X_true = (B0 + torch.matmul(Z, B))
    true_values = {
        "ZZt": ZZt_true,
        "P": proba_true,
        "Theta_X": Theta_X_true,
        "Theta_A": A_logit,
        "BBt": torch.mm(B.t(), B),
        "alpha": alpha
    }
    return true_values

def choose_init(init_method, K, Z, alpha, B, train):
    # initial values
    if init_method == "usvt_glm":
        init = initialize(train, K)
    elif init_method == "Btrue":
        init = {"weight": B}
    elif init_method == "Ztrue":
        init = {"positions": {"mean": Z}}
    elif init_method == "Btrue_Ztrue":
        init = {
            "weight": B,
            "positions": {"mean": Z}
        }
    elif init_method == "Btrue_Ztrue_alphatrue":
        init = {
            "weight": B,
            "positions": {"mean": Z},
            "heterogeneity": {"mean": alpha}
        }
    elif init_method == "Ztrue_Bglm":
        glm = GLM(train.K, train.N, train.p_cts, train.p_bin, mnar=False, latent_positions=Z)
        glm.fit(train, None, eps=1.e-6, max_iter=200, lr=1.)
        init = {
            "positions": {"mean": Z},
            "weight": glm.model.mean_model.weight.data.t()
        }
    else:  # defaults to random initialization
        init = dict()
    return init

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

# {('train', 'grad_Linfty'): 0.00021435308703523438,
#  ('train', 'grad_L1'): 0.03594383669776434,
#  ('train', 'grad_L2'): 0.0013316982829889312,
#  ('train', 'loss'): 7837.765197183397,
#  ('train', 'mse'): 0.9801869509349804,
#  ('train', 'auc'): 0.8806055805836732,
#  ('train', 'auc_A'): 0.9455831916095072,
#  ('test', 'loss'): 6118.997421871443,
#  ('test', 'mse'): 1.1178065883470472,
#  ('test', 'auc'): 0.8469045174164136,
#  ('test', 'auc_A'): 0.9455831916095072,
#  ('error', 'ZZt'): 0.5292599853387616,
#  ('error', 'P'): 0.02222798518916297,
#  ('error', 'Theta_X'): 0.05526458004207313,
#  ('error', 'Theta_A'): 0.11468221060356415,
#  ('error', 'BBt'): 0.9520764988373175,
#  ('error', 'alpha'): 0.16587158335799235,
#  ('train', 'time'): 116.3155312538147,
#  ('data', 'density'): 0.12130653266331658,
#  ('data', 'missing_prop'): 0.26749999999999996}


N = 50
K = 2
p_cts = 0
p_bin = 0
p = p_bin + p_cts
var_cov = 1.
missing_mean = -1000000.
seed = 7
alpha_mean_gen = -1.85
adjacency_noise = 0.
constant_components = False

K_model = 2
alpha_mean_model = -1.85
network_weight = 1.
estimate_components = False

algo = "VIMC"
# algo = "ADVI"
max_iter = 200
n_sample = 0
mcmc_n_sample = 2000
mcmc_n_chains = 5
mcmc_n_warmup = 1000
mcmc_n_thin = 10
optimizer = "Rprop"
eps = 0.0001
lr = 0.01
power = 0.
init_method = "random"
reg_B = 0. # 1. / (2. * 10.**2)
keep_logs = True

# MNAR
mnar_sparsity = 1.0
mnar_model = False
reg = 0.


# ---------------------------------------------------------------------
# generate data
Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0, W = \
    generate_dataset(
        N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cov, missing_mean=missing_mean,
        alpha_mean=alpha_mean_gen, seed=seed, mnar_sparsity=mnar_sparsity,
        adjacency_noise=adjacency_noise, constant_components=constant_components
    )
cuda = (algo in ["ADVI", "MLE", "MAP", "VIMC", "NetworkSmoothing"])
train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar_model, cuda=cuda)
test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar_model, test=True, cuda=cuda)
density = A.mean().item()
missing_prop = 0.
if X_cts is not None:
    missing_prop += X_cts.isnan().sum().item() / max(N * p, 1)
if X_bin is not None:
    missing_prop += X_bin.isnan().sum().item() / max(N * p, 1)
true_values = compute_true_values(B, B0, Z, alpha, i0, i1)
h_prior = (alpha_mean_model, 1.)
p_prior = (0., 1.)

# ---------------------------------------------------------------------
# initialize model
out = dict()
t0 = time.time()
if algo in ["ADVI", "MLE", "MAP", "VIMC"]:
    fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr, "reg": reg, "reg_B": reg_B,
                "power": power, "train": train, "test": test,
                "true_values": true_values, "return_log": True,
                "optimizer": optimizer}
    model_args = {"K": K_model, "N": N, "p_cts": p_cts, "p_bin": p_bin, "mnar": mnar_model,
                  "network_weight": network_weight,
                  "position_prior": p_prior, "heterogeneity_prior": h_prior,
                  "estimate_components": estimate_components}
    if algo == "ADVI":
        model = ADVI(**model_args)
    elif algo == "VIMC":
        fit_args["optimizer"] = "Adam"
        fit_args["lr"] *= 10.
        if n_sample == 0:  # 0 means default value
            n_sample = int(np.ceil(200/np.sqrt(max(N, p))))
        model = VIMC(n_samples=n_sample, **model_args)
    elif algo == "MLE":
        fit_args["optimizer"] = "Adam"
        fit_args["lr"] *= 10.
        model = MLE(**model_args)
    elif algo == "MAP":
        model = MAP(**model_args)
    init = choose_init(init_method, K, Z, alpha, B, train)
    model.init(**init)
    out = model.fit(**fit_args)
    if keep_logs:
        out, logs = out
elif algo == "MCMC":
    model = MCMC(K, N, p_cts, p_bin, (0., 1.), (alpha_mean_model, 1.))
    out = model.fit(
        train=train,
        n_sample=mcmc_n_sample,
        num_chains=mcmc_n_chains,
        num_warmup=mcmc_n_warmup,
        num_thin=mcmc_n_thin,
        true_values=true_values
    )
    diagnostics = model.diagnostic_summary()
    model.delete_fits()
elif algo in ["MICE", "Mean", "MissForest", "NetworkSmoothing"]:
    fit_args = {"train": train, "test": test}
    model_args = {"K": K_model, "N": N, "p_cts": p_cts, "p_bin": p_bin}
    if algo == "MICE":
        model = MICE(**model_args)
    if algo == "Mean":
        model = Mean(**model_args)
    if algo == "MissForest":
        model = MissForest(**model_args)
    if algo == "NetworkSmoothing":
        model = NetworkSmoothing(**model_args)
    out = model.fit(**fit_args)
else:
    raise ValueError("algorithm " + algo + " is not accepted for this experiment")

dt = time.time() - t0
out[("train", "time")] = dt
out[("data", "density")] = density
out[("data", "missing_prop")] = missing_prop


# fit_args["reg_B"] =  1. / (2. * 1.**2)
# out = model.fit(**fit_args)
# print(out[0][("test", "auc")])
# # mnar stuff
# out = model.fit_path(**fit_args)
#
# for r, (o, log) in out.items():
#     print(r, o[("test", "mse")])
#
#
# C.abs().sum(0) > 0
# model.covariate_weight[:p, :]
# B.T
# model.covariate_weight[p:, :]
# C.T
#
#
# BC = torch.hstack([B, C])
# W = model.covariate_weight.T
#
# torch.round(BC.T @ BC)
# torch.round(W.T @ W)
#
