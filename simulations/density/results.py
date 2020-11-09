import pandas as pd

SIM_NAME = "density"
SIM_PATH = "/home/simon/Documents/NNVI/simulations/" + SIM_NAME

results = pd.read_csv(SIM_PATH + "/results/summary.csv", index_col=0)

results.groupby("alpha_mean").agg("mean").to_csv(SIM_PATH + "/results/density.csv")