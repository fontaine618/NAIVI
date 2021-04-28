import sys
PATH = "/home/simfont/NAIVI/"
PATH = "/home/simon/Documents/NAIVI/results/"
sys.path.append(PATH)
import numpy as np
import torch
from sims.main import main

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH,
        name="covariate_continuous_simple",
        explore_dict={
            "data.N": np.array([100, 1000]),
            "data.K": np.array([5]),
            "data.p_bin": np.array([0]),
            "data.p_cts": np.array([5, 10, 20, 50, 100, 200, 500, 1000]),
            "data.missing_rate": np.array([0.1]),
            "data.seed": np.arange(0, 10, 1),
            "data.alpha_mean": np.array([-1.85]),
            "model.K": np.array([5]),
            "fit.algo": ["NetworkSmoothing", "Mean", "MissForest", ]
        }
    )
