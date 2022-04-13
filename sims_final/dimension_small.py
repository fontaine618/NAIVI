import sys
import os
import numpy as np
import torch

# PATH = "/home/simon/Documents/NAIVI/"
PATH = "/home/simfont/NAIVI/"
sys.path.append(PATH)
from NAIVI_experiments.main import main

# MCMC patch on greatlakes
os.environ["XDG_CACHE_HOME"] = "/home/simfont/scratch/.cache/"

if __name__ == "__main__":
    # from Slurm scheduler
    torch.set_default_dtype(torch.float64)
    SEED = int(os.getenv('SLURM_ARRAY_TASK_ID'))
    # GPU/CPU
    try:
        GPU = (int(sys.argv[1]) == 0)
    except IndexError:
        GPU = True
    NAME = "dimension_small" # should be dimension: remember this when doing graphs
    GPU_ALGOS = ["ADVI", "MAP", ] # with missing values
    CPU_ALGOS = ["MICE", "MissForest", "Mean"] # with missing values
    ALGOS = GPU_ALGOS if GPU else CPU_ALGOS
    WHICH = ("gpu" if GPU else "cpu") + "_seed" + str(SEED)
    main(
        path=PATH + "sims_final/results/",
        name=NAME,
        which=WHICH,
        explore_dict={
            "data.N": np.array([50]),
            "data.K": np.array([3, 5, 8]),
            "data.p_bin": np.array([100]),
            "data.p_cts": np.array([0]),
            "data.missing_mean": np.array([-1.]),
            "data.seed": np.array([SEED]),
            "data.alpha_mean": np.array([-1.85]),
            "data.mnar_sparsity": np.array([1.0]),
            "data.adjacency_noise": np.array([0.0]),
            "data.constant_components": [True],
            "model.K": np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]),
            "model.mnar": [False],
            "model.reg": np.array([0.0]),
            "model.alpha_mean": np.array([-1.85]),
            "model.network_weight": np.array([1.0]),
            "model.estimate_components": [False],
            "fit.algo": ALGOS,
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