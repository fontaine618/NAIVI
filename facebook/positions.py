import numpy as np
import torch
import matplotlib.pyplot as plt
from NAIVI_experiments.display import colormap, to_display
from matplotlib.patches import Ellipse
from matplotlib import collections  as mc
from facebook.data import get_data, get_featnames
from NAIVI.utils.data import JointDataset
from NAIVI.advi.model import ADVI

# setup
plt.style.use("seaborn")
PATH = "/home/simon/Documents/NAIVI/facebook/data/raw/"
OUT_PATH = "/home/simon/Documents/NAIVI/facebook/"
COLORS = colormap
DICT = to_display
color = "#5555ff"

# parametesr
center = 698
eps = 1.e-6
max_iter = 500
lr = 0.1
K_model = 2

#create dataset
i0, i1, A, X_cts, X_bin = get_data(PATH, center)
N = X_bin.size(0)
p_cts = 0
p_bin = X_bin.size(1)
p = p_cts + p_bin
train = JointDataset(i0, i1, A, X_cts, X_bin)

# fit model
model = ADVI(K_model, N, p_cts, p_bin)
fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr}
initial = {
    "bias": torch.zeros((1, p)).cuda(),
    "weight": torch.randn((K_model, p)).cuda(),
    "positions": torch.randn((N, K_model)).cuda(),
    "heterogeneity": torch.randn((N, 1)).cuda()*0.5 - 1.85
}
model.init(**initial)
output = model.fit(train, test=None, Z_true=None, **fit_args, batch_size=len(train))

# fit model without covaraites
train = JointDataset(i0, i1, A, None, None)
model_nocov = ADVI(K_model, N, 0, 0)
fit_args = {"eps": eps, "max_iter": max_iter, "lr": lr}
initial = {
    "positions": torch.randn((N, K_model)).cuda(),
    "heterogeneity": torch.randn((N, 1)).cuda()*0.5 - 1.85
}
model_nocov.init(**initial)
output = model_nocov.fit(train, test=None, Z_true=None, **fit_args, batch_size=len(train))


# extract w&b
weights = model.model.covariate_model.mean_model.weight.detach().cpu().numpy()
bias = model.model.covariate_model.mean_model.bias.detach().cpu().numpy()
weights_norm = (weights**2).sum(1)
top = np.argsort(weights_norm)[-6:][::-1]
featnames = get_featnames(PATH, center)

# get latent positions
mean_cov = model.model.encoder.latent_position_encoder.mean_encoder.values.detach().cpu().numpy()
stdev_cov = model.model.encoder.latent_position_encoder.log_var_encoder.values.exp().sqrt().detach().cpu().numpy()
mean_nocov = model_nocov.model.encoder.latent_position_encoder.mean_encoder.values.detach().cpu().numpy()
stdev_nocov = model_nocov.model.encoder.latent_position_encoder.log_var_encoder.values.exp().sqrt().detach().cpu().numpy()

# ======================================================================================================================
nrow = 2
ncol = 4
mult = np.sqrt(5.991)
X = X_bin.detach().cpu().numpy()
fig, axs = plt.subplots(nrow, ncol, figsize=(8, 4))
top = np.argsort(weights_norm)[-(nrow*ncol):][::-1]
for col, which in zip(range(ncol), top):
    for row in range(nrow):
        mean = mean_cov if row==0 else mean_nocov
        stdev = stdev_cov if row==0 else stdev_nocov
        ax = axs[row, col]
        for i in range(N):
            ellipse = Ellipse(xy=mean[i, ], width=mult*stdev[i, 0], height=mult*stdev[i, 1],
                              color=color, alpha=0.1)
            ax.add_patch(ellipse)
        for i, j, a in zip(i0.cpu().numpy(), i1.cpu().numpy(), A.cpu().numpy().flatten()):
            if (a==0.0):
                continue
            ax.plot(mean[(i, j), 0], mean[(i, j), 1], color=color, linewidth=1, alpha=0.2)
        ax.scatter(mean[:, 0], mean[:, 1], s=10,
                    color=["black" if X[i, which]==0. else "white" for i in range(N)],
                   edgecolors="black",
                   linewidth=1, zorder=100)
        if row==0:
            y = -(bias[which] + weights[which, 0])/weights[which, 1]
            x = -(bias[which] + weights[which, 1])/weights[which, 0]
            ax.axline((1., y), (x, 1.), color="black")
        ax.set_xlim(-4., 4.)
        ax.set_ylim(-4., 4.)
        if row==0:
            ax.set_title(f"{which}: {featnames.loc[which]}")
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.patch.set_facecolor('#EEEEEE')
axs[0, 0].set_ylabel("With attributes")
axs[1, 0].set_ylabel("Without attributes")
plt.tight_layout()
fig.savefig(OUT_PATH + f"figs/{center}_positions.pdf")
