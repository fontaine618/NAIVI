import pandas as pd
import numpy as np
from pypet import Trajectory
import matplotlib.pyplot as plt
plt.style.use('seaborn')

# load results
results = pd.read_csv(
	filepath_or_buffer="./simulations/missing_rate_bin/results/summary.csv",
	index_col=0
)

#
# # post processing redo
#
# traj = Trajectory("missing_rate_cts", add_time=False)
# traj.f_load(
# 	filename="./simulations/missing_rate_bin/results/missing_rate_bin.hdf5",
# 	load_results=2,
# 	load_data=2,
# 	load_parameters=2,
# 	load_other_data=2,
# 	load_derived_parameters=2
# )
# traj.v_auto_load = True
#
# p_bin = experiment["data.p_bin"]
# missing_rate = experiment["data.missing_rate"]
# seed = experiment["data.seed"]
#
# mse = [r.f_get("mse").mse for r in traj.res.runs]
# elbo = [r.f_get("elbo").elbo for r in traj.res.runs]
# auroc = [r.f_get("auroc").auroc for r in traj.res.runs]
# dist_inv = [r.f_get("dist_inv").dist_inv for r in traj.res.runs]
# dist_proj = [r.f_get("dist_proj").dist_proj for r in traj.res.runs]
# density = [r.f_get("density").density for r in traj.res.runs]
#
results = pd.DataFrame({
	"missing_rate": missing_rate,
	"seed": seed,
	"p_bin": p_bin,
	"elbo": elbo,
	"mse": mse,
	"auroc": auroc,
	"dist_inv": dist_inv,
	"dist_proj": dist_proj,
	"density": density
})

results.to_csv("./simulations/missing_rate_bin/results/summary.csv")


# drop stuff
results.dropna(inplace=True, how="any", axis=0)
results.drop(columns=["seed"], inplace=True)
# aggregate
stat = results.groupby(["p_bin", "missing_rate"]).agg(["mean", "std"])
n = results.groupby(["p_bin", "missing_rate"]).count()["elbo"]
stat["n"] = n

# statistics
statistics = {
	"elbo": "ELBO",
	"auroc": "AUROC",
	"dist_inv": "Distance to Z"
}
ps = {
	10: "darkred",
	100: "orangered",
	#500: "darkorange"
}

# plot
fig, axs = plt.subplots(3, 1, figsize=(4, 10), sharex=True)

for i, (st, name) in enumerate(statistics.items()):
	ax = axs[i]
	for p, col in ps.items():
		df = stat.loc[(p, )]
		mean = df[(st, "mean")]
		std = df[(st, "std")]
		n = df[("n")]
		se = std #/ np.sqrt(n)
		ax.plot(df.index, mean, color=col, label=p)
		ax.fill_between(df.index, mean-se, mean+se, color=col, alpha=0.2)
	if i == 2:
		plt.legend(loc="center right", title="Nb. covariates")
		ax.set_xlabel("Missing rate")
	ax.set_ylabel(name)

fig.tight_layout()
fig.savefig("./simulations/missing_rate_bin/results/statistics.pdf")

plt.close("all")