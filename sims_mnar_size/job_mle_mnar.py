import sys
PATH = "/home/simon/Documents/NAIVI/sims_mnar_size/"
sys.path.append(PATH)
import numpy as np
import torch
from sims_mnar.main import main

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH,
        name="mle_mnar",
        explore_dict={
            "data.N": np.array([50, 100, 200, 500]),
            "data.K": np.array([5]),
            "data.p_bin": np.array([0]),
            "data.p_cts": np.array([100]),
            "data.missing_mean": np.linspace(-3., 1., 13),
            "data.seed": np.arange(0, 10, 1),
            "data.alpha_mean": np.array([-1.85]),
            "data.mnar_sparsity": np.array([0.0]),
            "fit.algo": ["MLE"],
            "model.K": np.array([5]),
            "model.mnar": [True]
        }
    )