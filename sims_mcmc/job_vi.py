import sys
# PATH = "/home/simon/Documents/NAIVI/"
PATH = "/home/simfont/NAIVI/"
sys.path.append(PATH)
import numpy as np
import torch
from sims_mcmc.main import main

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH + "sims_mcmc/",
        name="vi",
        explore_dict={
            "data.N": np.array([25, 50, 100, 200, 500, 1000, 2000]),
            "data.K": np.array([2]),
            "data.p_bin": np.array([0]),
            "data.p_cts": np.array([0, 200]),
            "data.missing_mean": np.array([-10000.]),
            "data.seed": np.arange(0, 10, 1),
            "data.alpha_mean": np.array([-1.85]),
            "data.mnar_sparsity": np.array([0.0]),
            "fit.algo": ["VIMC", "ADVI", "MLE", "MAP"],
            "fit.n_sample": np.array([0]),
            "fit.mcmc_n_sample": np.array([1000]),
            "model.alpha_mean": np.array([-1.85]),
            "model.K": np.array([2]),
            "model.mnar": [False]
        }
    )
