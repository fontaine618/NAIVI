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
    torch.set_default_dtype(torch.float64)
    GPU = True
    NAME = "estimation_N"
    GPU_ALGOS = ["VIMC", "ADVI", "MAP"]
    CPU_ALGOS = ["MCMC"]
    # CPU_ALGOS = ["MICE", "MissForest", "Mean", "NetworkSmoothing"]
    ALGOS = GPU_ALGOS if GPU else CPU_ALGOS
    NAME = NAME + ("_gpu" if GPU else "_cpu")
    main(
        path=PATH + "/results/",
        name=NAME,
        explore_dict={
            "data.N": np.array([30]), # np.array([25, 50, 100, 200, 500, 1000]),  # Vary N
            "data.K": np.array([2]),
            "data.p_bin": np.array([0]),
            "data.p_cts": np.array([5]), # np.array([0, 50]),  # 2 sub-experiments
            "data.missing_mean": np.array([-1.]),
            "data.seed": np.arange(0, 1, 1), # np.arange(0, 10, 1),  # 10 replications
            "data.alpha_mean": np.array([-1.85]),
            "data.mnar_sparsity": np.array([0.0]),
            "data.adjacency_noise": np.array([0.0]),
            "data.constant_components": [True],
            "model.K": np.array([2]),
            "model.mnar": [False],
            "model.reg": np.array([0.0]),
            "model.alpha_mean": np.array([-1.85]),
            "model.network_weight": np.array([1.0]),
            "model.estimate_components": [False],
            "fit.algo": ALGOS,
            "fit.max_iter": np.array([200]),
            "fit.n_sample": np.array([0]),
            "fit.mcmc_n_sample": np.array([1000]),
            "fit.mcmc_n_chains": np.array([5]),
            "fit.mcmc_n_warmup": np.array([500]),
            "fit.lr": np.array([0.001]),
            "fit.eps": np.array([1e-8]),
            "fit.power": np.array([0.0]),
            "fit.init": ["random"],
            "fit.optimizer": ["Rprop"],
        }
    )