import sys
import os
import numpy as np
import torch

# PATH = "/home/simon/Documents/NAIVI/"
PATH = "/home/simfont/NAIVI/"
sys.path.append(PATH)
from NAIVI_facebook.main import main

if __name__ == "__main__":
    # from Slurm scheduler
    torch.set_default_dtype(torch.float64)
    SEED = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    NAME = "fb_dimension"
    WHICH = "seed" + str(SEED)
    main(
        path=PATH + "sims_facebook/results/",
        name=NAME,
        which=WHICH,
        explore_dict={
            "data.center": np.array([0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]),
            "data.missing_prop": np.array([0.5]),
            "data.seed": np.array([SEED]),
            "model.K": np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]),
            "model.mnar": [False],
            "model.reg": np.array([0.0]),
            "model.alpha_mean": np.array([-1.85]),
            "model.network_weight": np.array([1.0]),
            "model.estimate_components": [False],
            "fit.algo": ["ADVI"],
            "fit.max_iter": np.array([200]),
            "fit.n_sample": np.array([0]),
            "fit.mcmc_n_sample": np.array([2000]),
            "fit.mcmc_n_chains": np.array([5]),
            "fit.mcmc_n_warmup": np.array([1000]),
            "fit.lr": np.array([0.01]),
            "fit.eps": np.array([0.0005]),
            "fit.power": np.array([0.0]),
            "fit.init": ["random"],
            "fit.optimizer": ["Rprop"],
        }
    )