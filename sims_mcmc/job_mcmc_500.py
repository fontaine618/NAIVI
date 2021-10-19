import sys
# PATH = "/home/simon/Documents/NAIVI/"
PATH = "/home/simfont/NAIVI/"
sys.path.append(PATH)
import numpy as np
import torch
from sims_mcmc.main import main


if __name__ == "__main__":
    if len(sys.argv) > 1:
        which = int(sys.argv[1])
        seed = np.array([which % 10])
        ps = np.array([[0, 50][which // 10]])
        name = f"mcmc_500_{which}"
    else: # no argument = run all
        seed = np.arange(0, 10, 1)
        ps = np.array([0, 50])
        name = "mcmc_500_all"
    print(seed, ps)
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH + "sims_mcmc/",
        name=name,
        explore_dict={
            "data.N": np.array([500]),
            "data.K": np.array([2]),
            "data.p_bin": np.array([0]),
            "data.p_cts": ps,
            "data.missing_mean": np.array([-10000.]),
            "data.seed": seed,
            "data.alpha_mean": np.array([-1.85]),
            "data.mnar_sparsity": np.array([0.0]),
            "fit.algo": ["MCMC"],
            "fit.n_sample": np.array([0]),
            "fit.mcmc_n_sample": np.array([1000]),
            "model.alpha_mean": np.array([-1.85]),
            "model.K": np.array([2]),
            "model.mnar": [False]
        }
    )
