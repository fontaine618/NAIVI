import sys
import os
PATH = "/home/simon/Documents/NAIVI/"
if not os.path.isdir(PATH):
    # patch for great lakes
    PATH = "/home/simfont/NAIVI/"
sys.path.append(PATH)
import numpy as np
import torch
from NAIVI_experiments.main import main

NAME = "missing_rate"

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH + "results/",
        name=NAME,
        which="all",
        explore_dict={
            "data.N": np.array([200, ]),
            "data.K": np.array([3]),
            "data.p_bin": np.array([100, ]),
            "data.p_cts": np.array([0, ]),
            "data.missing_mean": np.linspace(-3., 3., 13),
            "data.seed": np.arange(0, 10, 1),
            "data.alpha_mean": np.array([-1.85]),
            "data.mnar_sparsity": np.array([1.0]),
            "data.var_cov": np.array([1.0]),
            "model.alpha_mean": np.array([-1.85]),
            "model.K": np.array([3]),
            "model.mnar": [False],
            "fit.algo": ["VMP", "ADVI", "MAP", ],
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
