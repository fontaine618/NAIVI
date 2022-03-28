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


def run(traj):
    # ---------------------------------------------------------------------
    # extract generating parameters
    N = int(traj.par.data.N)
    K = int(traj.par.data.K)
    p_cts = int(traj.par.data.p_cts)
    p_bin = int(traj.par.data.p_bin)
    p = p_bin + p_cts
    var_cov = traj.par.data.var_cov
    missing_mean = traj.par.data.missing_mean
    seed = int(traj.par.data.seed)
    alpha_mean_gen = traj.par.data.alpha_mean
    mnar_sparsity = traj.par.data.mnar_sparsity
    adjacency_noise = traj.par.data.adjacency_noise
    constant_components = traj.par.data.constant_components
    # extract model parameters
    K_model = int(traj.par.model.K)
    mnar = traj.par.model.mnar
    alpha_mean_model = traj.par.model.alpha_mean
    reg = traj.par.model.reg
    network_weight = traj.par.model.network_weight
    estimate_components = traj.par.model.estimate_components
    # extract fit parameters
    algo = traj.par.fit.algo
    max_iter = int(traj.par.fit.max_iter)
    n_sample = int(traj.par.fit.n_sample)
    mcmc_n_sample = int(traj.par.fit.mcmc_n_sample)
    mcmc_n_chains = int(traj.par.fit.mcmc_n_chains)
    mcmc_n_warmup = int(traj.par.fit.mcmc_n_warmup)
    mcmc_n_thin = int(traj.par.fit.mcmc_n_thin)
    optimizer = traj.par.fit.optimizer
    eps = traj.par.fit.eps
    lr = traj.par.fit.lr
    keep_logs = traj.par.fit.keep_logs
    power = traj.par.fit.power
    init_method = traj.par.fit.init
    print_settings(traj)
    try:
        # ---------------------------------------------------------------------
        # generate data
        Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0, W = \
        generate_dataset(
            N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cov, missing_mean=missing_mean,
            alpha_mean=alpha_mean_gen, seed=seed, mnar_sparsity=mnar_sparsity,
            adjacency_noise=adjacency_noise, constant_components=constant_components
        )
        cuda = (algo in ["ADVI", "MLE", "MAP", "VIMC", "NetworkSmoothing"])
        train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar, cuda=cuda)
        test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar,
                            test=True, cuda=cuda)
        density = A.mean().item()
        missing_prop = 0.
        if X_cts is not None:
            missing_prop += X_cts.isnan().sum().item() / max(N*p, 1)
        if X_bin is not None:
            missing_prop += X_bin.isnan().sum().item() / max(N*p, 1)
        true_values = compute_true_values(B, B0, Z, alpha, i0, i1)
        h_prior = (alpha_mean_model, 1.)
        p_prior = (0., 1.)
        # ---------------------------------------------------------------------
        # initialize model
        out = dict()
        t0 = time.time()
        if algo in ["ADVI", "MLE", "MAP", "VIMC"]:
            fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr, "reg": reg,
                        "power": power, "train": train, "test": test,
                        "true_values": true_values, "return_log": True,
                        "optimizer": optimizer}
            model_args = {"K": K_model, "N": N, "p_cts": p_cts, "p_bin": p_bin, "mnar": mnar,
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
                model = MLE(**model_args)
            elif algo == "MAP":
                model = MAP(**model_args)
            init = choose_init(init_method, K, Z, alpha, B, train)
            model.init(**init)
            out = model.fit(**fit_args)
            if keep_logs:
                out, logs = out
                traj.f_add_result("logs.$", df=logs)
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
            traj.f_add_result("diagnostics.$", df=diagnostics)
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

        # ---------------------------------------------------------------------
        # fit model
        dt = time.time() - t0
        out[("train", "time")] = dt
        out[("data", "density")] = density
        out[("data", "missing_prop")] = missing_prop
        for k, v in out.items():
            traj.f_add_result("values.$.{}.{}".format(*k), v)
        for k, v in out.items():
            print(f"{k[0]:<10} {k[1]:<16} {v:4f}")
        return out
    except Exception as e:
        traj.f_add_result("error_message.$", e)
        return None


def print_settings(traj):
    # print for display
    print("=" * 80)
    print("RUN SETTINGS\n")
    print("-" * 80)
    print("DATA SETTINGS")
    for v in traj.par.data:
        if isinstance(v, Parameter):
            print("{:<20} {}".format(v.v_name, v._data))
    print("-" * 80)
    print("MODEL SETTINGS")
    for v in traj.par.model:
        if isinstance(v, Parameter):
            print("{:<20} {}".format(v.v_name, v._data))
    print("-" * 80)
    print("FIT SETTINGS")
    for v in traj.par.fit:
        if isinstance(v, Parameter):
            print("{:<20} {}".format(v.v_name, v._data))
    print("=" * 80)


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