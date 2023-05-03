import torch
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pypet_experiments.facebook import get_data, get_featnames
from NAIVI.vmp import VMP
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


torch.set_default_tensor_type(torch.cuda.FloatTensor)
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


# Get data
center = 698
path = "./datasets/facebook/"
i0, i1, A, X_cts, X_bin = get_data(path, center)
featnames = get_featnames(path, center)


# prepare stuff
n = max(i0.max().item(), i1.max().item()) + 1
adj = torch.eye(n)
adj[i0, i1] = A.flatten().float()
degrees = adj.mean(0)
hmean = torch.logit(degrees).mean().item()
hvar = torch.logit(degrees).var().item()*1.5


# Joint LSM
jlsm = VMP(
    latent_dim=2,
    n_nodes=X_bin.shape[0],
    heterogeneity_prior_mean=hmean,
    heterogeneity_prior_variance=hvar,
    latent_prior_mean=0.,
    latent_prior_variance=1.,
    binary_covariates=X_bin.float(),
    continuous_covariates=X_cts,
    edges=A.float(),
    edge_index_left=i0,
    edge_index_right=i1,
)
jlsm.fit(max_iter=1000, rel_tol=1e-5)



# LSM
lsm = VMP(
    latent_dim=2,
    n_nodes=X_bin.shape[0],
    heterogeneity_prior_mean=hmean,
    heterogeneity_prior_variance=hvar,
    latent_prior_mean=0.,
    latent_prior_variance=1.,
    binary_covariates=None, #X_bin.float(),
    continuous_covariates=X_cts,
    edges=A.float(),
    edge_index_left=i0,
    edge_index_right=i1,
)
lsm.fit(max_iter=1000, rel_tol=1e-5)



# FA
fa = VMP(
    latent_dim=2,
    n_nodes=X_bin.shape[0],
    heterogeneity_prior_mean=hmean,
    heterogeneity_prior_variance=hvar,
    latent_prior_mean=0.,
    latent_prior_variance=1.,
    binary_covariates=X_bin.float(),
    continuous_covariates=X_cts,
    edges=None, #A.float(),
    edge_index_left=i0,
    edge_index_right=i1,
)
fa.fit(max_iter=1000, rel_tol=1e-5)


models = {
    "Network & Attributes": jlsm,
    "Network only": lsm,
    "Attributes only": fa,
}


# Plot
variables = [ 9, 28, 38, 39]
variables = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

nrows = len(models)
ncols = len(variables)

plt.cla()
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2.5, nrows*2.5),
                         sharex=True, sharey=True)
for row, (model_name, model) in enumerate(models.items()):
    Zm, Zv = model.variables["latent"].posterior.mean_and_variance
    bias = model.bias.flatten()
    weights = model.weights  # K x p
    for col, var in enumerate(variables):
        ax = axes[row, col]
        # add lines at 0
        ax.axhline(y=0, color="black", linewidth=1, zorder=1, linestyle="dashed")
        ax.axvline(x=0, color="black", linewidth=1, zorder=1, linestyle="dashed")
        if row == 0:
            ax.set_title(featnames[var], fontsize=12)
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        # add edges
        for i, j, a in zip(i0, i1, A):
            if a == 0:
                continue
            ax.plot(
                [Zm[i, 0].item(), Zm[j, 0].item()],
                [Zm[i, 1].item(), Zm[j, 1].item()],
                color="#aaaaaa", linewidth=0.5,
                alpha=0.2
            )
        # add positions
        x = X_bin[:, var].float()
        c = ["#1111ff" if xi == 0 else "#ff1111" for xi in x]
        m = ["o" if xi == 0 else "s" for xi in x]
        ax.scatter(
            Zm[:, 0].cpu(), Zm[:, 1].cpu(),
            s=10, alpha=1,
            color=c, zorder=100,
        )
        # add decision boundary
        if model_name != "Network only":
            w = weights[:, var].cpu()
            b = bias[var].item()
            x = torch.linspace(-4, 4, 100).cpu()
            y = -(w[0]*x + b)/w[1]
            ax.plot(x, y, color="black", linewidth=1)
    axes[row, 0].set_ylabel(model_name, labelpad=10, fontsize=12)
    axes[row, 0].set_yticks([-4, -2, 0, 2, 4])
plt.tight_layout()
plt.savefig("./experiments/facebook/position_plot.pdf", bbox_inches="tight")
