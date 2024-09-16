import torch
from NAIVI.mcmc import MCMC
from NAIVI.vmp import VMP
import matplotlib.pyplot as plt
import seaborn as sns


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

experiments = {
    # name: (Display, n_nodes, p_cts, p_bin, latent_dim)
    "test": ("Test", 20, 5, 0, 3),
}

results = dict()

for name, (display, n_nodes, p_cts, p_bin, latent_dim) in experiments.items():
    # generate data
    torch.manual_seed(0)
    p = p_bin + p_cts
    w = torch.randn(latent_dim, p)
    b = torch.randn(1, p)
    Z = torch.randn(n_nodes, latent_dim)
    a = torch.randn(n_nodes) - 1.85
    thetaX = Z @ w + b
    mean_cts, logit_bin = thetaX[:, :p_cts], thetaX[:, p_cts:]
    X_cts = mean_cts + torch.randn_like(mean_cts)
    X_bin = torch.bernoulli(torch.sigmoid(logit_bin))
    M = torch.randint(0, 2, (n_nodes, p))
    M_cts, M_bin = M[:, :p_cts], M[:, p_cts:]
    X_cts_missing = torch.where(M_cts == 1, torch.full_like(X_cts, float("nan")), X_cts)
    X_bin_missing = torch.where(M_bin == 1, torch.full_like(X_bin, float("nan")), X_bin)
    X_cts = torch.where(M_cts == 0, torch.full_like(X_cts, float("nan")), X_cts)
    X_bin = torch.where(M_bin == 0, torch.full_like(X_bin, float("nan")), X_bin)
    i = torch.tril_indices(n_nodes, n_nodes, offset=-1)
    i0, i1 = i[0, :], i[1, :]
    thetaA = (Z[i0, :] * Z[i1, :]).sum(1) + a[i0] + a[i1]
    pA = torch.sigmoid(thetaA)
    A = torch.bernoulli(pA)
    i_cts, j_cts = (~torch.isnan(X_cts)).nonzero(as_tuple=True)
    i_bin, j_bin = (~torch.isnan(X_bin)).nonzero(as_tuple=True)
    i_cts_missing, j_cts_missing = (torch.isnan(X_cts_missing)).nonzero(as_tuple=True)
    i_bin_missing, j_bin_missing = (torch.isnan(X_bin_missing)).nonzero(as_tuple=True)
    # priors
    h = {
        "latent_prior_variance": 1.,
        "latent_prior_mean": 0.,
        "heterogeneity_prior_variance": 1.,
        "heterogeneity_prior_mean": -1.85,
        "loading_prior_variance": 100.,
        "cts_variance": (10., 10.)
    }
    # fit VMP
    vmp = VMP(
        latent_dim=latent_dim,
        n_nodes=n_nodes,
        binary_covariates=X_bin,
        continuous_covariates=X_cts,
        edges=A.unsqueeze(1),
        edge_index_left=i0,
        edge_index_right=i1,
        **h
    )
    vmp.fit()
    vmp_output = vmp.output_with_uncertainty()
    # fit MCMC
    mcmc = MCMC(
        n_nodes=n_nodes,
        latent_dim=latent_dim,
        binary_covariates=X_bin,
        continuous_covariates=X_cts,
        edges=A.float(),
        edge_index_left=i0,
        edge_index_right=i1,
        **h
    )
    mcmc.fit()
    mcmc_output = mcmc.output_with_uncertainty()
    # store results
    res = dict(vmp=dict(), mcmc=dict())
    res["vmp"]["X_cts_mean_error"] = (vmp_output["pred_continuous_covariates"][0] - X_cts)[i_cts_missing, j_cts_missing]
    res["mcmc"]["X_cts_mean_error"] = (mcmc_output["pred_continuous_covariates"][0] - X_cts)[i_cts_missing, j_cts_missing]

    plt.cla()
    plt.scatter(res["mcmc"]["X_cts_mean_error"].pow(2.), res["vmp"]["X_cts_mean_error"].pow(2.))
    plt.axline((0, 0), (1, 1), color="black", linestyle="--")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()



