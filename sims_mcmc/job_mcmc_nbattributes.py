import sys
# PATH = "/home/simon/Documents/NAIVI/"
PATH = "/home/simfont/NAIVI/"
sys.path.append(PATH)
import numpy as np
import torch
from sims_mcmc.main import main

PS = [0, 10, 20, 50, 100, 200, 500, 1000]  # 8
# NS = [200, 1000] # 2


if __name__ == "__main__":
    if len(sys.argv) > 1:
        which = int(sys.argv[1])
        seed = which % 10
        exp = which // 10
        n = 200
        p = PS[exp]
        name = f"mcmc_p{p}_n{n}_{seed}"
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH + "sims_mcmc/",
        name=name,
        explore_dict={
            "data.N": np.array([n]),
            "data.K": np.array([2]),
            "data.p_bin": np.array([0]),
            "data.p_cts": np.array([p]),
            "data.missing_mean": np.array([-10000.]),
            "data.seed": np.array([seed]),
            "data.alpha_mean": np.array([-1.85]),
            "data.mnar_sparsity": np.array([0.0]),
            "fit.algo": ["MCMC"],
            "fit.n_sample": np.array([0]),
            "fit.mcmc_n_sample": np.array([5000]),
            "fit.mcmc_n_chains": np.array([10]),
            "fit.mcmc_n_warmup": np.array([1000]),
            "model.alpha_mean": np.array([-1.85]),
            "model.K": np.array([2]),
            "model.mnar": [False]
        }
    )
