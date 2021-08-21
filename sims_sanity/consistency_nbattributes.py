import sys
PATH = "/home/simon/Documents/NAIVI/sims_sanity/"
sys.path.append(PATH)
import numpy as np
import torch
from sims_sanity.main import main

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH,
        name="r_consistency_nbattributes_VIMC50",
        explore_dict={
            "data.N": np.array([200]),
            "data.K": np.array([2]),
            "data.p_bin": np.array([0]),
            "data.p_cts": np.array([0, 10, 20, 50, 100, 200, 500, 1000]),
            "data.missing_mean": np.array([-100.]),
            "data.seed": np.arange(0, 10, 1),
            "data.alpha_mean": np.array([-1.85]),
            "data.mnar_sparsity": np.array([0.0]),
            # "fit.algo": ["VIMC", "ADVI", "MLE"],
            "fit.algo": ["VIMC", ],
            "fit.n_sample": np.array([50]),
            "model.K": np.array([2]),
            "model.mnar": [False]
        }
    )
