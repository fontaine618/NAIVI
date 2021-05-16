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
        name="Kfb_simple",
        explore_dict={
            "data.missing_rate": np.array([0.50]),
            "data.seed": np.arange(0, 10, 1),
            "data.center": np.array([698, 0, 1684]),
            "model.K": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]),
            "fit.algo": ["NetworkSmoothing", "Mean", "MissForest"],
            "fit.max_iter": np.array([1000])
        }
    )