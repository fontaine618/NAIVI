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
plt.style.use("seaborn-whitegrid")
PATH = "/home/simon/Documents/NAIVI/facebook/data/raw/"
OUT_PATH = "/home/simon/Documents/NAIVI/facebook/"
COLORS = colormap
DICT = to_display
torch.manual_seed(0)

# parametesr
center = 698
eps = 0.0005
max_iter = 2000
lr = 0.01
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
output = model.fit(train, test=None, **fit_args)

# extract w&b
weights = model.model.covariate_model.mean_model.weight.detach().cpu().numpy()
bias = model.model.covariate_model.mean_model.bias.detach().cpu().numpy()
weights_norm = (weights**2).sum(1)
# get latent positions
mean_cov = model.model.encoder.latent_position_encoder.mean_encoder.values.detach().cpu().numpy()
stdev_cov = model.model.encoder.latent_position_encoder.log_var_encoder.values.exp().sqrt().detach().cpu().numpy()

# fit model without network
model_nonet = ADVI(K_model, N, p_cts, p_bin, network_weight=0.)
output_nonet = model_nonet.fit(train, test=None, **fit_args)
# extract w&b
weights_nonet = model_nonet.model.covariate_model.mean_model.weight.detach().cpu().numpy()
bias_nonet = model_nonet.model.covariate_model.mean_model.bias.detach().cpu().numpy()
# get latent positions
mean_nonet = model_nonet.model.encoder.latent_position_encoder.mean_encoder.values.detach().cpu().numpy()
stdev_nonet = model_nonet.model.encoder.latent_position_encoder.log_var_encoder.values.exp().sqrt().detach().cpu().numpy()

# fit model without covaraites
train = JointDataset(i0, i1, A, None, None)
model_nocov = ADVI(K_model, N, 0, 0)
output_nocov = model_nocov.fit(train, test=None, **fit_args)
# get latent positions
mean_nocov = model_nocov.model.encoder.latent_position_encoder.mean_encoder.values.detach().cpu().numpy()
stdev_nocov = model_nocov.model.encoder.latent_position_encoder.log_var_encoder.values.exp().sqrt().detach().cpu().numpy()


# ======================================================================================================================
color = "#5599ff"
nrow = 3
ncol = 4
mult = np.sqrt(5.991)
X = X_bin.detach().cpu().numpy()
plt.cla()
fig, axs = plt.subplots(nrow, ncol, figsize=(8, 6))
top = np.argsort(weights_norm)[-(nrow*ncol):][::-1]
top = [3, 28, 38, 39] #2, 3, 6, 7, 9, 15, 28, 38, 39 # 7,
featnames = get_featnames(PATH, center)
for col, which in zip(range(ncol), top):
    for row in range(nrow):
        mean = [mean_cov, mean_nocov, mean_nonet][row]
        stdev = [stdev_cov, stdev_nocov, stdev_nonet][row]
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
        if row == 0:
            y = -(bias[which] + weights[which, 0])/weights[which, 1]
            x = -(bias[which] + weights[which, 1])/weights[which, 0]
            ax.axline((1., y), (x, 1.), color="black")
        if row == 2:
            y = -(bias_nonet[which] + weights_nonet[which, 0])/weights_nonet[which, 1]
            x = -(bias_nonet[which] + weights_nonet[which, 1])/weights_nonet[which, 0]
            ax.axline((1., y), (x, 1.), color="black")
        ax.set_xlim(-4., 4.)
        ax.set_ylim(-4., 4.)
        if row == 0:
            ax.set_title(f"{which}: {featnames.loc[which]}")
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
axs[0, 0].set_ylabel("With attributes and network")
axs[1, 0].set_ylabel("Without attributes")
axs[2, 0].set_ylabel("Without network")
plt.tight_layout()
fig.savefig(OUT_PATH + f"figs/{center}_positions.pdf")




# ======================================================================================================================
color = "#5599ff"
nrow = 1
ncol = 3
mult = np.sqrt(5.991)
X = X_bin.detach().cpu().numpy()
fig, axs = plt.subplots(nrow, ncol, figsize=(6, 2))
top = [20]
featnames = get_featnames(PATH, center)
for row, which in zip(range(nrow), top):
    for col in range(ncol):
        mean = [mean_cov, mean_nocov, mean_nonet][col]
        stdev = [stdev_cov, stdev_nocov, stdev_nonet][col]
        ax = axs[col]
        # for i in range(N):
        #     ellipse = Ellipse(xy=mean[i, ], width=mult*stdev[i, 0], height=mult*stdev[i, 1],
        #                       color=color, alpha=0.1)
        #     ax.add_patch(ellipse)
        for i, j, a in zip(i0.cpu().numpy(), i1.cpu().numpy(), A.cpu().numpy().flatten()):
            if (a==0.0):
                continue
            ax.plot(mean[(i, j), 0], mean[(i, j), 1], color=color, linewidth=1, alpha=0.2)
        ax.scatter(mean[:, 0], mean[:, 1], s=10,
                    color=["black" if X[i, which]==0. else "white" for i in range(N)],
                   edgecolors="black",
                   linewidth=1, zorder=100)
        if col == 0:
            y = -(bias[which] + weights[which, 0])/weights[which, 1]
            x = -(bias[which] + weights[which, 1])/weights[which, 0]
            ax.axline((1., y), (x, 1.), color="black")
        if col == 2:
            y = -(bias_nonet[which] + weights_nonet[which, 0])/weights_nonet[which, 1]
            x = -(bias_nonet[which] + weights_nonet[which, 1])/weights_nonet[which, 0]
            ax.axline((1., y), (x, 1.), color="black")
        ax.set_xlim(-3., 3.)
        ax.set_ylim(-3., 3.)
        if col == 0:
            ax.set_ylabel(f"{which}: {featnames.loc[which]}")
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        # colors
        ax.patch.set_facecolor('#ffffff')
        for line in ax.get_xgridlines():
            # line.set_color("#CCCCCC")
            line.set_color("#FFFFFF")
            line.set_linewidth(0.5)
        for line in ax.get_ygridlines():
            # line.set_color("#CCCCCC")
            line.set_color("#FFFFFF")
            line.set_linewidth(0.5)
        for axis in ax.spines.values():
            axis.set_color("white")
            axis.set_linewidth(2)
axs[0].set_title("With attributes and network")
axs[1].set_title("Without attributes")
axs[2].set_title("Without network")


plt.rcParams['savefig.facecolor'] = (1, 1, 1, 0)

plt.tight_layout()
fig.savefig(OUT_PATH + f"figs/{center}_positions_poster.png", transparent=True)



