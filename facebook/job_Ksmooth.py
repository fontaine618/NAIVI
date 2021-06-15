import sys
PATH = "/home/simon/Documents/NAIVI/facebook/"
sys.path.append(PATH)
import numpy as np
import torch
from facebook.main import main
from facebook.data import get_centers

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH,
        name="Kfb_smooth",
        explore_dict={
            "data.missing_rate": np.array([0.50]),
            "data.seed": np.arange(0, 10, 1),
            "data.center": np.array([698, 0, 1684]),
            "model.K": np.array([1]),
            "fit.algo": ["NetworkSmoothing", ],
            "fit.max_iter": np.array([1000])
        }
    )