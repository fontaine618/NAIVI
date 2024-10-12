import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


plt.rcParams.update(plt.rcParamsDefault)
sns.set_theme(style="whitegrid", palette="colorblind")
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.default": "regular",
    "font.family": ["sans-serif"],
    "font.sans-serif": ["Lato"],
    "axes.labelweight": "normal",
    "figure.titleweight": "bold",
    "figure.titlesize": "large",
    "font.weight": "normal",
    # "axes.formatter.use_mathtext": True,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

EXPERIMENT_NAME = "vmp_v_mcmc"
DIR_RESULTS = f"./experiments/{EXPERIMENT_NAME}/results/"
DIR_FIGURES = f"./experiments/{EXPERIMENT_NAME}/"

experiments = {
    # name: (Display, n_nodes, p_cts, p_bin, latent_dim)
    "50_5_3": ("A", 50, 5, 0, 3),
    "50_20_3": ("B", 50, 20, 0, 3),
    "100_5_5": ("C", 100, 5, 0, 5),
    "100_20_5": ("D", 100, 20, 0, 5),
}

results = dict()

for name, _ in experiments.items():
    # load results
    with open(f"{DIR_RESULTS}{name}.json", "r") as f:
        res = json.load(f)
    results[name] = res

columns = experiments
rows = {
    # name: (Display, stat, obs, missing)
    "pred_bias": ("Pred. Bias", "bias", "X_cts", "X_cts_missing"),
    "pred_sd": ("Pred. Std. Dev.", "var", "X_cts", "X_cts_missing"),
    "pred_rmse": ("Pred. RMSE", "mse", "X_cts", "X_cts_missing"),
    "linpred_bias": ("Lin. Pred. Bias", "bias", "X_cts", "X_cts_missing_fitted"),
    "linpred_sd": ("Lin. Pred. Std. Dev.", "var", "X_cts", "X_cts_missing_fitted"),
    "linpred_rmse": ("Lin. Pred. RMSE", "mse", "X_cts", "X_cts_missing_fitted"),
    "logitA_bias": ("Logit Edge Prob Bias", "bias", "thetaA", None),
    "logitA_sd": ("Logit Edge Prob Std. Dev.", "var", "thetaA", None),
    "logitA_rmse": ("Logit Edge Prob RMSE", "mse", "thetaA", None),
}

fig, axes = plt.subplots(nrows=len(rows), ncols=len(columns), figsize=(20, 40))
for r, (rname, (rdisp, rstat, robs, rmiss)) in enumerate(rows.items()):
    for c, (cname, (cdisp, n_nodes, p_cts, p_bin, latent_dim)) in enumerate(columns.items()):
        if rstat == "bias":
            x = np.array(results[cname]["mcmc"][f"{robs}_bias"])
            y = np.array(results[cname]["vmp"][f"{robs}_bias"])
            x = np.abs(x)
            y = np.abs(y)
            if rmiss is not None:
                xmiss = np.array(results[cname]["mcmc"][f"{rmiss}_bias"])
                ymiss = np.array(results[cname]["vmp"][f"{rmiss}_bias"])
                xmiss = np.abs(xmiss)
                ymiss = np.abs(ymiss)
        elif rstat == "var":
            x = np.array(results[cname]["mcmc"][f"{robs}_var"])
            y = np.array(results[cname]["vmp"][f"{robs}_var"])
            x = np.sqrt(x)
            y = np.sqrt(y)
            if rmiss is not None:
                xmiss = np.array(results[cname]["mcmc"][f"{rmiss}_var"])
                ymiss = np.array(results[cname]["vmp"][f"{rmiss}_var"])
                xmiss = np.sqrt(xmiss)
                ymiss = np.sqrt(ymiss)
        elif rstat == "rmse":
            x = np.array(results[cname]["mcmc"][f"{robs}_bias"])**2 + np.array(results[cname]["mcmc"][f"{robs}_var"])
            y = np.array(results[cname]["vmp"][f"{robs}_bias"])**2 + np.array(results[cname]["vmp"][f"{robs}_var"])
            x = np.sqrt(x)
            y = np.sqrt(y)
            if rmiss is not None:
                xmiss = np.array(results[cname]["mcmc"][f"{rmiss}_bias"])**2 + np.array(results[cname]["mcmc"][f"{rmiss}_var"])
                ymiss = np.array(results[cname]["vmp"][f"{rmiss}_bias"])**2 + np.array(results[cname]["vmp"][f"{rmiss}_var"])
                xmiss = np.sqrt(xmiss)
                ymiss = np.sqrt(ymiss)
        xlabel = "MCMC"
        ylabel = "VMP"
        title = f"{cdisp}: {rdisp}"
        maxval = max(x.max(), y.max())
        if rmiss is not None:
            maxval = max(maxval, xmiss.max(), ymiss.max())
        axes[r, c].set_xlim(0, maxval)
        axes[r, c].set_ylim(0, maxval)
        axes[r, c].axline((0, 0), (1, 1), color="black", linestyle="--", alpha=0.5)
        if rmiss is not None:
            axes[r, c].scatter(xmiss, ymiss, alpha=0.5, label="Missing", color="blue")
        axes[r, c].scatter(x, y, alpha=0.5, label="Observed", color="red")
        axes[r, c].set_title(title)
        axes[r, c].set_xlabel(xlabel)
        axes[r, c].set_ylabel(ylabel)
        axes[r, c].set_aspect("equal")
plt.tight_layout()
plt.savefig(f"{DIR_FIGURES}comparison.png")


