import numpy as np
import time
from NAIVI_experiments.gen_data import generate_dataset
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
    missing_rate = traj.par.data.missing_rate
    seed = traj.par.data.seed
    alpha_mean = traj.par.data.alpha_mean
    # extract model parameters
    K_model = traj.par.model.K
    # extract fit parameters
    algo = traj.par.fit.algo
    max_iter = traj.par.fit.max_iter
    n_sample = traj.par.fit.n_sample
    eps = traj.par.fit.eps
    lr = traj.par.fit.lr
    # print
    print("="*80)
    print(seed, N, p_cts, p_bin, K, missing_rate, alpha_mean, K_model, algo)
    input = seed, N, p_cts, p_bin, K, missing_rate, alpha_mean, K_model, algo
    try:
        # ---------------------------------------------------------------------
        # generate data
        Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0 = generate_dataset(
            N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cov, missing_rate=missing_rate,
            alpha_mean=alpha_mean, seed=seed
        )
        train = JointDataset(i0, i1, A, X_cts, X_bin)
        test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing)
        density = A.mean().item()
        # initial values
        initial = {
            "bias": B0.cuda(),
            "weight": B.cuda(),
            "positions": Z.cuda(),
            "heterogeneity": alpha.cuda()
        }
        # ---------------------------------------------------------------------
        # initialize model
        if algo == "ADVI":
            model = ADVI(K_model, N, p_cts, p_bin)
            fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr}
        elif algo == "VIMC":
            model = VIMC(K_model, N, p_cts, p_bin)
            fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr, "n_sample": n_sample}
        elif algo == "VMP":
            raise RuntimeError("VMP not implemented yet")
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
            fit_args = {}
        else:
            model = MLE(K, N, p_cts, p_bin)
            fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr}
        # set initial values
        model.init(**initial)

        # ---------------------------------------------------------------------
        # fit model
        t0 = time.time()
        output = model.fit(train, test, Z.cuda(), **fit_args, batch_size=len(train)) + [density, time.time() - t0]
    except RuntimeError as e:
        print(e)
        output = [np.nan for _ in range(11)]
    finally:
        print(*output)
        return input, output