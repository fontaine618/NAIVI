import numpy as np
import time
import torch
from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI.advi.model import ADVI
from NAIVI.mle.model import MLE
from NAIVI.vimc.model import VIMC
from NAIVI.mice.model import MICE
from NAIVI.mf import MissForest
from NAIVI.constant import Mean
from NAIVI.smoothing import NetworkSmoothing


def run(traj):
    # ---------------------------------------------------------------------
    # extract generating paramters
    N = traj.par.data.N
    K = traj.par.data.K
    p_cts = traj.par.data.p_cts
    p_bin = traj.par.data.p_bin
    var_cov = traj.par.data.var_cov
    missing_mean = traj.par.data.missing_mean
    seed = traj.par.data.seed
    alpha_mean = traj.par.data.alpha_mean
    mnar_sparsity = traj.par.data.mnar_sparsity
    # extract model parameters
    K_model = traj.par.model.K
    mnar = traj.par.model.mnar
    # extract fit parameters
    algo = traj.par.fit.algo
    max_iter = traj.par.fit.max_iter
    n_sample = traj.par.fit.n_sample
    eps = traj.par.fit.eps
    lr = traj.par.fit.lr
    reg = 10**np.linspace(1., -1., 10)
    # print
    print("="*80)
    print(seed, N, p_cts, p_bin, K, missing_mean, alpha_mean, K_model, mnar, algo, mnar_sparsity)
    input = seed, N, p_cts, p_bin, K, missing_mean, alpha_mean, K_model, mnar, algo, mnar_sparsity
    try:
        # ---------------------------------------------------------------------
        # generate data
        Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0 = generate_dataset(
            N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cov, missing_mean=missing_mean,
            alpha_mean=alpha_mean, seed=seed, mnar_sparsity=mnar_sparsity
        )
        train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar)
        test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar, test=True)
        density = A.mean().item()
        missing_rate = 0.
        if X_bin is not None:
            missing_rate += np.isnan(X_bin).sum().item()
        if X_cts is not None:
            missing_rate += np.isnan(X_cts).sum().item()
        missing_rate /= N * (p_bin+p_cts)
        # initial values
        if mnar:
            B0 = torch.cat([B0, C0], 1)
            B = torch.cat([B, C], 1)
        init = {"positions": Z, "heterogeneity": alpha, "bias": B0, "weight": B}
        # ---------------------------------------------------------------------
        # initialize model
        if algo == "ADVI":
            model = ADVI(K_model, N, p_cts, p_bin, mnar=mnar)
            fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr,
                        "batch_size": len(train), "reg": reg, "init": init}
        elif algo == "VIMC":
            model = VIMC(K_model, N, p_cts, p_bin, mnar=mnar, n_samples=n_sample)
            fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr,
                        "batch_size": len(train), "reg": reg, "init": init}
        elif algo =="MICE":
            model = MICE(K_model, N, p_cts, p_bin)
            fit_args = {}
        elif algo =="MissForest":
            model = MissForest(K_model, N, p_cts, p_bin)
            fit_args = {}
        elif algo =="Mean":
            model = Mean(K_model, N, p_cts, p_bin)
            fit_args = {}
        elif algo =="NetworkSmoothing":
            model = NetworkSmoothing(K_model, N, p_cts, p_bin)
            fit_args = {"batch_size": len(train)}
        else:
            model = MLE(K, N, p_cts, p_bin, mnar=mnar)
            fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr,
                        "batch_size": len(train), "reg": reg, "init": init}
        # set initial values
        model.init(**init)

        # ---------------------------------------------------------------------
        # fit model
        t0 = time.time()
        if algo in ["MLE", "ADVI", "VIMC"] and mnar:
            _, _, out = model.fit_path(train, test, Z_true=Z.cuda(), **fit_args)
            if out is not None:
                output, selected = out[:-1], out[-1]
                true = torch.where(B.abs().sum(0) > 0., 1, 0).numpy()
                acc = ((true==selected)[(p_bin+p_cts):]).mean()
                output += [acc]
            else:
                output = [np.nan for _ in range(14)]
        if algo in ["MLE", "ADVI", "VIMC"] and not mnar:
            del fit_args["init"]
            fit_args["reg"] = 0.
            output = model.fit(train, test, Z_true=Z.cuda(), **fit_args)
            if out is not None:
                output += [0., 0.]
            else:
                output = [np.nan for _ in range(14)]
        else:
            output = model.fit(train, test, Z_true=Z.cuda(), **fit_args)
            output += [0, 0.] # number of non-zero  and accuracy not applicable here
        output += [density, missing_rate, time.time() - t0]
    except RuntimeError as e:
        print(e)
        output = [np.nan for _ in range(17)]
    finally:
        print(*output)
        return input, output