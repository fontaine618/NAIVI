import sys
PATH = "/home/simfont/NNVI/"
sys.path.append(PATH)
import numpy as np
import torch
from sims.main import main

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH,
        name="missingrate_binary_mice",
        explore_dict={
            "data.N": np.array([500]),
            "data.K": np.array([5]),
            "data.p_cts": np.array([0]),
            "data.p_bin": np.array([10, 100, 500]),
            "data.missing_rate": np.array([0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]),
            "data.seed": np.arange(0, 10, 1),
            "data.alpha_mean": np.array([-1.85]),
            "model.K": np.array([5]),
            "fit.algo": ["MICE"]
        }
    )
