import pandas as pd
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from pypet import Trajectory
from matplotlib.lines import Line2D
from NAIVI_experiments.display import colormap, to_display

plt.style.use("seaborn")
RESULTS_PATH = "/home/simon/Documents/NAIVI/sims_convergence/results/"
FIGS_PATH = "/home/simon/Documents/NAIVI/sims_convergence/figs/"

# load data
traj = Trajectory("optimizer")
traj.f_load(filename=RESULTS_PATH + "optimizer.hdf5", force=True)
traj.v_auto_load = True
results = traj.res.summary.results.df
parameters = traj.res.summary.parameters.df
logs = traj.res.summary.logs.df

# Prepare stuff
optimizers = parameters[("fit", "optimizer")]
cols = [matplotlib.colors.to_hex(col) for col in sns.color_palette("Set2", len(optimizers))]
colors = {
    name: col for name, col in zip(optimizers, cols)
}
metrics = {
    ("train", "loss"): "(Training loss - min)/min",
    ("train", "grad_Linfty"): "Linfty(grad)",
    ("train", "grad_L1"): "L1(grad)",
    ("train", "grad_L2"): "L2(grad)",
    ("train", "mse"): "(Training MSE - min)/min",
    ("train", "auc_A"): "(max - Training AUC)",
    ("error", "BBt"): "MSE(BBt)",
    ("error", "Theta_X"): "MSE(Theta_X)",
    ("error", "ZZt"): "MSE(ZZt)",
    ("error", "alpha"): "MSE(alpha)",
    ("error", "Theta_A"): "MSE(Theta_A)",
    ("error", "P"): "MSE(P)",
}
min_loss = logs[("train", "loss")].min()
min_mse =logs[("train", "mse")].min()
max_auc = logs[("train", "auc_A")].max()

# Plot
fig, axs = plt.subplots(len(metrics), 1, figsize=(8, 20), sharey="row", sharex="col")
for ax, (metric, display) in zip(axs, metrics.items()):
    for name, col in colors.items():
        i = optimizers.index[optimizers == name]
        m = logs.loc[(i, slice(None)), metric]
        m.index = m.index.droplevel(0)
        if metric == ("train", "loss"):
            m = (m - min_loss) / min_loss
        if metric == ("train", "mse"):
            m = (m - min_mse) / min_mse
        if metric == ("train", "auc_A"):
            m = max_auc - m
        ax.plot(m, color=col)
    ax.set_yscale("log")
    ax.set_ylabel(display)
axs[0].set_xscale("log")

# legend
lines = [Line2D([0], [0], color=col, linestyle="-")
         for _, col in colors.items()]
labels = colors.keys()
fig.legend(lines, labels, loc=8, ncol=len(colors)//2)
fig.tight_layout()
fig.subplots_adjust(bottom=0.05)
fig.savefig(FIGS_PATH + "optimizer.pdf")