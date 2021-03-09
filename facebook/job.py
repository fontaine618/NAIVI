import sys
PATH = "//facebook/"
sys.path.append(PATH)
import numpy as np
import torch
from facebook.main import main
from facebook.data import get_centers

centers = get_centers(PATH + "data/raw/")

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(
        path=PATH,
        name="Kfb_1684_mle",
        explore_dict={
            "data.missing_rate": np.array([0.50]),
            "data.seed": np.arange(0, 10, 1),
            "data.center": np.array([698, 0, 1684]),
            "model.K": np.array([12, 15, 20]),
            "fit.algo": ["MLE"],
            "fit.max_iter": np.array([1000])
        }
    )
