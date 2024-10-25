import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


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


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

columns = experiments
rows = {
    # name: (Display, stat, obs, missing)
    "pred_bias": ("Pred. bias", "bias", "X_cts", "X_cts_missing"),
    "pred_sd": ("Pred. std. dev.", "var", "X_cts", "X_cts_missing"),
    "pred_rmse": ("Pred. RMSE", "mse", "X_cts", "X_cts_missing"),
    # "linpred_bias": ("Lin. Pred. Bias", "bias", "X_cts", "X_cts_missing_fitted"),
    "linpred_sd": ("Lin. Pred. Std. Dev.", "var", "X_cts", "X_cts_missing_fitted"),
    "linpred_rmse": ("Lin. Pred. RMSE", "mse", "X_cts", "X_cts_missing_fitted"),
    "logitA_bias": ("Logit Edge Prob Bias", "bias", "thetaA", None),
    "logitA_sd": ("Logit Edge Prob Std. Dev.", "var", "thetaA", None),
    # "logitA_rmse": ("Logit Edge Prob RMSE", "mse", "thetaA", None),
}

fig, axes = plt.subplots(nrows=len(rows), ncols=len(columns), figsize=(2.5*len(columns)-1., 3*4+1))
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
        elif rstat == "mse":
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
        maxval = max(x.max(), y.max())
        if rmiss is not None:
            maxval = max(maxval, xmiss.max(), ymiss.max())
        axes[r, c].set_xlim(0, maxval)
        axes[r, c].set_ylim(0, maxval)
        axes[r, c].axline((0, 0), (1, 1), color="black", linestyle="--", alpha=0.5, zorder=-1)
        if rmiss is not None:
            axes[r, c].scatter(xmiss, ymiss, alpha=0.1, label="Missing", color="blue", s=10, marker="s")
        axes[r, c].scatter(x, y, alpha=0.1, label="Observed", color="red", s=10, marker="o")
        if rmiss is not None:
            confidence_ellipse(xmiss, ymiss, axes[r, c], n_std=2, edgecolor="white", alpha=1.0,  linewidth=4)
            confidence_ellipse(xmiss, ymiss, axes[r, c], n_std=2, edgecolor="blue", alpha=1.0, linewidth=2)
        confidence_ellipse(x, y, axes[r, c], n_std=2, edgecolor="white", alpha=1.0, linewidth=4)
        confidence_ellipse(x, y, axes[r, c], n_std=2, edgecolor="red", alpha=1.0, linewidth=2)
        if rmiss is not None:
            axes[r, c].scatter(xmiss.mean(), ymiss.mean(), color="white", s=100, marker="s")
            axes[r, c].scatter(xmiss.mean(), ymiss.mean(), color="blue", s=50, marker="s")
        axes[r, c].scatter(x.mean(), y.mean(), color="white", s=100, marker="o")
        axes[r, c].scatter(x.mean(), y.mean(), color="red", s=50, marker="o")
        if r==0:
            title = f"N={n_nodes}, P={p_cts}, K={latent_dim}"
            axes[r, c].set_title(title)
        if r==len(rows)-1:
            axes[r, c].set_xlabel(xlabel)
        if c==0:
            axes[r, c].set_ylabel(ylabel)
        if c==len(columns)-1:
            # ylabel on the other side
            axes[r, c].yaxis.set_label_position("right")
            axes[r, c].set_ylabel(rdisp, rotation=270, labelpad=15)
        axes[r, c].set_aspect("equal")
plt.tight_layout()
plt.savefig(f"{DIR_FIGURES}comparison.png")


