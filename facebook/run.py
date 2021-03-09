import numpy as np
import torch
import time
from facebook.data import get_data
from NAIVI.utils.data import JointDataset
from NAIVI.advi.model import ADVI
from NAIVI.mle.model import MLE
from NAIVI.vimc.model import VIMC
from NAIVI.mice.model import MICE

PATH = "//facebook/data/raw/"
# PATH = "/home/simfont/NAIVI/facebook/data/raw/"

# centers = get_centers(PATH)
#
# for node in centers:
#     i0, i1, A, X_cts, X_bin = get_data(PATH, node)
#     print(X_bin.size())

def run(traj):
    # extract parameters
    center = traj.par.data.center
    missing_rate = traj.par.data.missing_rate
    seed = traj.par.data.seed
    K = traj.par.data.K
    alpha_mean = traj.par.data.alpha_mean
    # extract model parameters
    K_model = traj.par.model.K
    # extract fit parameters
    algo = traj.par.fit.algo
    max_iter = traj.par.fit.max_iter
    n_sample = traj.par.fit.n_sample
    eps = traj.par.fit.eps
    lr = traj.par.fit.lr
    # get data
    i0, i1, A, X_cts, X_bin = get_data(PATH, center)
    # recover data
    N = X_bin.size(0)
    p_cts = 0
    p_bin = X_bin.size(1)
    p = p_cts + p_bin
    density = A.mean().item()
    # insert missing values
    torch.manual_seed(seed)
    mask = torch.rand_like(X_bin) < missing_rate
    X_bin_missing = torch.where(~mask, np.nan, X_bin)
    X_bin = torch.where(mask, np.nan, X_bin)
    X_cts_missing = None
    # print
    print("="*80)
    print(center, seed, N, p_cts, p_bin, K, missing_rate, alpha_mean, K_model, algo)
    input = center, seed, N, p_cts, p_bin, K, missing_rate, alpha_mean, K_model, algo
    # dataset format
    train = JointDataset(i0, i1, A, X_cts, X_bin)
    test = JointDataset(i0, i1, A, X_cts_missing, X_bin_missing)
    # initialization
    initial = {
        "bias": torch.zeros((1, p)).cuda(),
        "weight": torch.randn((K_model, p)).cuda(),
        "positions": torch.randn((N, K_model)).cuda(),
        "heterogeneity": torch.randn((N, 1)).cuda()*0.5 - 1.85
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
    elif algo == "MICE":
        model = MICE(K_model, N, p_cts, p_bin)
        fit_args = {}
    else:
        model = MLE(K_model, N, p_cts, p_bin)
        fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr}
    # set initial values
    model.init(**initial)
    try:
        # ---------------------------------------------------------------------
        # fit model
        t0 = time.time()
        output = model.fit(train, test, None, **fit_args, batch_size=len(train)) + [density, time.time() - t0]
    except RuntimeError as e:
        print(e)
        output = [np.nan for _ in range(11)]
    finally:
        print(*output)
        return input, output
