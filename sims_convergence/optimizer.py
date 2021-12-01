import sys
PATH = "/home/simon/Documents/NAIVI/"
# PATH = "/home/simfont/NAIVI/"
sys.path.append(PATH)
import numpy as np
import torch
from sims_convergence.main import main

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH + "sims_convergence/",
        name="power_lr",
        explore_dict={
            "data.N": np.array([200, ]),
            "data.K": np.array([5]),
            "data.p_bin": np.array([0]),
            "data.p_cts": np.array([1000]),
            "data.missing_mean": np.array([-10000.]),
            "data.seed": np.arange(0, 1, 1),
            "data.alpha_mean": np.array([-1.85]),
            "data.mnar_sparsity": np.array([0.0]),
            "fit.algo": ["ADVI", ],
            "fit.n_sample": np.array([0]),
            "fit.max_iter": np.array([5000]),
            "fit.lr": np.array([0.01]),
            "fit.power": np.array([0., ]),
            "fit.optimizer": ["Adam", "SGD", "Adagrad", "Adadelta", "AdamW", "Adamax", "ASGD", "RMSprop", "Rprop"],
            "model.alpha_mean": np.array([-1.85]),
            "model.K": np.array([5]),
            "model.mnar": [False]
        }
    )
