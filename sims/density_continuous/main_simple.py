import sys
PATH = "/home/simfont/NAIVI/"
sys.path.append(PATH)
import numpy as np
import torch
from sims.main import main

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH,
        name="density_continuous_simple",
        explore_dict={
            "data.N": np.array([500]),
            "data.K": np.array([5]),
            "data.p_bin": np.array([0]),
            "data.p_cts": np.array([10, 100, 500]),
            "data.missing_rate": np.array([0.1]),
            "data.seed": np.arange(0, 10, 1),
            "data.alpha_mean": np.array([-3.2, -2.8, -2.4, -2., -1.6, -1.2, -0.8, -0.4, 0.0, 0.4]),
            "model.K": np.array([5]),
            "fit.algo": ["NetworkSmoothing", "Mean", "MissForest", ]
        }
    )
