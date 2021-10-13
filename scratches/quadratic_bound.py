import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import itertools as it
import pandas as pd
plt.style.use("seaborn")

PATH = "/home/simon/Documents/NAIVI/sims_sanity/"

A = 0.
mu = 10.
sigma = 1.


def mc_estimate(A, mu, sigma, n_sample=1000):
    logits = np.random.normal(mu, sigma, n_sample)
    prob = 1. / (1. + np.exp(-logits))
    prob = prob.clip(1.e-6, 1.-1.e-6)
    llk = A * np.log(prob) + (1. - A) * np.log(1. - prob)
    return np.mean(llk)

def quadratic_bound(A, mu, t):
    bound = - np.log(1. + np.exp(-t)) + (A - 0.5) * mu - 0.5 * t
    return bound


def best_quadratic_bound(A, mu, sigma):
    t = np.sqrt(mu**2 + sigma**2)
    return quadratic_bound(A, mu, t)


mus = np.linspace(-5., 5., 21)
sigmas = 10 ** np.linspace(-2, 1., 4)
sigmas = [2., 1., 0.5, 0.1]
exp = list(it.product(mus, sigmas))

out = pd.DataFrame(columns=["mu", "sigma", "mc0", "mc1", "qb0", "qb1"])
for i, (mu, sigma) in enumerate(exp):
    mc0 = mc_estimate(0., mu, sigma, 1000000)
    mc1 = mc_estimate(1., mu, sigma, 1000000)
    qb0 = best_quadratic_bound(0., mu, sigma)
    qb1 = best_quadratic_bound(1., mu, sigma)
    out.loc[i] = (mu, sigma, mc0, mc1, qb0, qb1)

out.set_index(["sigma", "mu"], inplace=True)



colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
for i, sigma in enumerate(sigmas):
    ax.plot(mus, np.abs(out.loc[(sigma, ), "mc0"]), color=colors[i],
            label=f"{sigma} (true)", linestyle="-", linewidth=1)
    ax.plot(mus, np.abs(out.loc[(sigma, ), "qb0"]), color=colors[i],
            label=f"{sigma} (bound)", linestyle="--", linewidth=1)
# ax.set_yscale("log")
ax.set_xlabel("$\mu$")
ax.set_ylabel("Value")
ax.set_yscale("log")
# legend
fig.legend(loc=8, ncol=4)

fig.tight_layout() #h_pad=0.5, w_pad=0.)
fig.subplots_adjust(bottom=0.30)

fig.savefig(PATH + "figs/quadratic_bound.pdf")


pip install numpy==1.19.5 pandas==1.1.4 pystan==3.1.1 scipy==1.5.3 scikit-learn==0.23.2 torch==1.7.0 pypet==0.5.1 missingpy==0.2.0 pytorch_lightning==1.0.6