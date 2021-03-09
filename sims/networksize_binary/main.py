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
        name="networksize_binary",
        explore_dict={
            "data.N": np.array([50, 100, 200, 500, 1000, 2000]),
            "data.K": np.array([5]),
            "data.p_cts": np.array([0]),
            "data.p_bin": np.array([10, 100, 500]),
            "data.missing_rate": np.array([0.10]),
            "data.seed": np.arange(0, 10, 1),
            "data.alpha_mean": np.array([-1.85]),
            "model.K": np.array([5]),
            "fit.algo": ["MLE", "ADVI", "VIMC"]
        }
    )
