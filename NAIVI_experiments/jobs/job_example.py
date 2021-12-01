import sys
import os
PATH = "//"
if not os.path.isdir(PATH):
    # patch for great lakes
    PATH = "/home/simfont/NAIVI/"
sys.path.append(PATH)
import numpy as np
import torch
from NAIVI_experiments.main import main

NAME = "example"

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH + "results/",
        name=NAME,
        explore_dict={
            "data.N": np.array([50, ]),
            "data.K": np.array([5]),
            "data.p_bin": np.array([0]),
            "data.p_cts": np.array([20, ]),
            "data.missing_mean": np.array([-10000.]),
            "data.seed": np.arange(0, 1, 1),
            "data.alpha_mean": np.array([-1.85]),
            "data.mnar_sparsity": np.array([1.0]),
            "data.var_cov": np.array([1.0]),
            "model.alpha_mean": np.array([-1.85]),
            "model.K": np.array([5]),
            "model.mnar": [False],
            "fit.algo": ["ADVI", "VIMC", "MLE", "MAP", "MCMC", "MissForest", "MICE", "Mean", "NetworkSmoothing"],
            "fit.n_sample": np.array([0]),
            "fit.max_iter": np.array([500]),
            "fit.mcmc_n_sample": np.array([100]),
            "fit.mcmc_n_warmup": np.array([100]),
            "fit.mcmc_n_chains": np.array([10]),
            "fit.lr": np.array([0.01]),
            "fit.eps": np.array([1e-5]),
            "fit.optimizer": ["Rprop"],
            "fit.power": np.array([0., ]),
            "fit.init": ["random"]
        }
    )
