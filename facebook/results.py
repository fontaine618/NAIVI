import pandas as pd
import os

PATH = "//facebook/"

dir = os.listdir(PATH)
folders = [x for x in dir if x.find(".") < 0]
exps = [x for x in folders if x[:2] == "fb"]

results = pd.concat([
    pd.read_csv("{}{}/results/summary.csv".format(PATH, ex), index_col=0)
    for ex in exps
])

summary = results.groupby(["missing_rate", "center", "algo"]).agg("mean")
summary.to_csv(PATH + "results/summary.csv")