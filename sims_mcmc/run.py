import numpy as np
import time
import torch
from NAIVI_experiments.gen_data_mnar import generate_dataset
from NAIVI.utils.data import JointDataset
from NAIVI.advi import ADVI
from NAIVI.mle import MLE
from NAIVI.vimc import VIMC
from NAIVI.mcmc import MCMC


def run(traj):
    # ---------------------------------------------------------------------
    # extract generating paramters
    N = int(traj.par.data.N)
    K = int(traj.par.data.K)
    p_cts = int(traj.par.data.p_cts)
    p_bin = int(traj.par.data.p_bin)
    var_cov = traj.par.data.var_cov
    missing_mean = traj.par.data.missing_mean
    seed = int(traj.par.data.seed)
    alpha_mean_gen = traj.par.data.alpha_mean
    mnar_sparsity = traj.par.data.mnar_sparsity
    # extract model parameters
    K_model = int(traj.par.model.K)
    mnar = traj.par.model.mnar
    alpha_mean_model = traj.par.model.alpha_mean
    # extract fit parameters
    algo = traj.par.fit.algo
    max_iter = int(traj.par.fit.max_iter)
    n_sample = int(traj.par.fit.n_sample)
    mcmc_n_sample = int(traj.par.fit.mcmc_n_sample)
    eps = traj.par.fit.eps
    lr = traj.par.fit.lr
    reg = 0.
    # print
    print("="*80)
    print(seed, N, p_cts, p_bin, K, missing_mean, alpha_mean_gen, K_model, mnar, algo, mnar_sparsity, n_sample)
    input = seed, N, p_cts, p_bin, K, missing_mean, alpha_mean_gen, K_model, mnar, algo, mnar_sparsity, n_sample
    try:
        # ---------------------------------------------------------------------
        # generate data
        Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, i0, i1, A, B, B0, C, C0 = generate_dataset(
            N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_cov=var_cov, missing_mean=missing_mean,
            alpha_mean=alpha_mean_gen, seed=seed, mnar_sparsity=mnar_sparsity
        )
        cuda = (algo != "MCMC")
        print(f"algo={algo}, cuda={cuda}")
        train = JointDataset(i0, i1, A, X_cts, X_bin, return_missingness=mnar, cuda=cuda)
        test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing, return_missingness=mnar,
                            test=True, cuda=cuda)
        density = A.mean().item()
        E = i0.shape[0]
        p = p_bin + p_cts

        # prepare true calues for comparison
        ZZt_true = (Z @ Z.T).detach().cpu().numpy()
        A_logit = alpha[i0] + alpha[i1] + torch.sum(Z[i0, :] * Z[i1, :], 1, keepdim=True)
        proba_true = torch.sigmoid(A_logit).detach().cpu().numpy()
        Theta_X_true = (B0 + torch.matmul(Z, B)).detach().cpu().numpy()
        # ---------------------------------------------------------------------
        # initialize model
        if algo == "ADVI":
            model = ADVI(K_model, N, p_cts, p_bin, mnar=mnar)
            fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr,
                        "batch_size": len(train), "reg": reg}
        elif algo == "VIMC":
            if n_sample == 0:  # 0 means default value
                n_sample = int(np.ceil(200/np.sqrt(N)))
            model = VIMC(K_model, N, p_cts, p_bin, mnar=mnar, n_samples=n_sample)
            fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr,
                        "batch_size": len(train), "reg": reg}
        elif algo == "MLE":
            model = MLE(K, N, p_cts, p_bin, mnar=mnar)
            fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr,
                        "batch_size": len(train), "reg": reg}
        elif algo == "MCMC":
            model = MCMC(K, N, p_cts, p_bin, (0., 1.), (alpha_mean_model, 1.))
        else:
            raise ValueError("algorithm " + algo + " is not accepted for this experiment")

        # ---------------------------------------------------------------------
        # fit model
        t0 = time.time()
        if algo in ["MLE", "ADVI", "VIMC"]:
            model.fit(train, test, Z_true=Z, alpha_true=alpha, **fit_args)
            dt = time.time() - t0
            Z_est = model.latent_positions()
            alpha_est = model.latent_heterogeneity()
            ZZt_est = (Z_est @ Z_est.T).detach().numpy()
            A_logit = alpha_est[i0] + alpha_est[i1] + torch.sum(Z_est[i0, :] * Z_est[i1, :], 1, keepdim=True)
            proba_est = torch.sigmoid(A_logit).detach().numpy()
            B0_est = model.model.covariate_model.bias
            B_est = model.model.covariate_model.weight.T
            if p > 0:
                Theta_X_est = (B0_est + torch.matmul(Z_est, B_est)).detach().numpy()
        elif algo == "MCMC":
            print("Stan model: " + model.model)
            model.fit(train, max_iter=mcmc_n_sample)
            dt = time.time() - t0
            ZZt_est = model.posterior_mean("ZZt")
            proba_est = model.posterior_mean("proba").reshape((-1, 1))
            if p > 0:
                Theta_X_est = model.posterior_mean("Theta_X")
        else:  # ["Mean", "NetworkSmoothing", "MICE", "MissForest"]
            raise ValueError("algorithm " + algo + " is not accepted for this experiment")
        DZ = ((ZZt_est - ZZt_true)**2).sum() / (ZZt_true**2).sum()
        DP = ((proba_est - proba_true)**2).sum() / (proba_true**2).sum()
        if p > 0:
            DThetaX = ((Theta_X_est - Theta_X_true)**2).sum() / (ZZt_true**2).sum()
        else:
            DThetaX = 0.
        output = (DZ, DP, DThetaX, dt)
    except RuntimeError as e:
        print(e)
        output = (np.nan for _ in range(4))
    finally:
        print(*output)
        return input, output