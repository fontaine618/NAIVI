import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
import os
import json
import sys
sys.path.insert(1, '/home/simon/Documents/NAIVI/')
from NAIVI.mcmc import MCMC
from NAIVI.vmp import VMP
# torch.set_default_tensor_type(torch.cuda.FloatTensor)


EXPERIMENT_NAME = "vmp_v_mcmc"
DIR_RESULTS = "./results/"
if not os.path.exists(DIR_RESULTS):
    os.makedirs(DIR_RESULTS)

experiments = {
    # name: (Display, n_nodes, p_cts, p_bin, latent_dim)
    "50_5_3": ("A", 50, 5, 0, 3),
    "50_20_3": ("B", 50, 20, 0, 3),
    "100_5_5": ("C", 100, 5, 0, 5),
    "100_20_5": ("D", 100, 10, 0, 5),
}

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
    # prediction bias
    res["vmp"]["X_cts_bias"] = (vmp_output["pred_continuous_covariates"][0] - X_cts)[i_cts, j_cts].cpu().numpy().tolist()
    res["mcmc"]["X_cts_bias"] = (mcmc_output["pred_continuous_covariates"][0] - X_cts)[i_cts, j_cts].cpu().numpy().tolist()
    res["vmp"]["X_cts_missing_bias"] = (vmp_output["pred_continuous_covariates"][0] - X_cts_missing)[i_cts_missing, j_cts_missing].cpu().numpy().tolist()
    res["mcmc"]["X_cts_missing_bias"] = (mcmc_output["pred_continuous_covariates"][0] - X_cts_missing)[i_cts_missing, j_cts_missing].cpu().numpy().tolist()
    # prediction variance
    res["vmp"]["X_cts_var"] = vmp_output["pred_continuous_covariates"][1][i_cts, j_cts].cpu().numpy().tolist()
    res["mcmc"]["X_cts_var"] = mcmc_output["pred_continuous_covariates"][1][i_cts, j_cts].cpu().numpy().tolist()
    res["vmp"]["X_cts_missing_var"] = vmp_output["pred_continuous_covariates"][1][i_cts_missing, j_cts_missing].cpu().numpy().tolist()
    res["mcmc"]["X_cts_missing_var"] = mcmc_output["pred_continuous_covariates"][1][i_cts_missing, j_cts_missing].cpu().numpy().tolist()
    # fitted mean bias
    res["vmp"]["X_cts_fitted_bias"] = (vmp_output["linear_predictor_covariates"][0] - thetaX)[i_cts, j_cts].cpu().numpy().tolist()
    res["mcmc"]["X_cts_fitted_bias"] = (mcmc_output["linear_predictor_covariates"][0] - thetaX)[i_cts, j_cts].cpu().numpy().tolist()
    res["vmp"]["X_cts_missing_fitted_bias"] = (vmp_output["linear_predictor_covariates"][0] - thetaX)[i_cts_missing, j_cts_missing].cpu().numpy().tolist()
    res["mcmc"]["X_cts_missing_fitted_bias"] = (mcmc_output["linear_predictor_covariates"][0] - thetaX)[i_cts_missing, j_cts_missing].cpu().numpy().tolist()
    # fitted mean variance
    res["vmp"]["X_cts_fitted_var"] = vmp_output["linear_predictor_covariates"][1][i_cts, j_cts].cpu().numpy().tolist()
    res["mcmc"]["X_cts_fitted_var"] = mcmc_output["linear_predictor_covariates"][1][i_cts, j_cts].cpu().numpy().tolist()
    res["vmp"]["X_cts_missing_fitted_var"] = vmp_output["linear_predictor_covariates"][1][i_cts_missing, j_cts_missing].cpu().numpy().tolist()
    res["mcmc"]["X_cts_missing_fitted_var"] = mcmc_output["linear_predictor_covariates"][1][i_cts_missing, j_cts_missing].cpu().numpy().tolist()
    # fitted logit edge bias
    res["vmp"]["thetaA_bias"] = (vmp_output["linear_predictor_edges"][0].squeeze(1) - thetaA).cpu().numpy().tolist()
    res["mcmc"]["thetaA_bias"] = (mcmc_output["linear_predictor_edges"][0] - thetaA).cpu().numpy().tolist()
    # fitted logit edge variance
    res["vmp"]["thetaA_var"] = vmp_output["linear_predictor_edges"][1].squeeze(1).cpu().numpy().tolist()
    res["mcmc"]["thetaA_var"] = mcmc_output["linear_predictor_edges"][1].cpu().numpy().tolist()
    # store results as json
    with open(f"{DIR_RESULTS}/{name}.json", "w") as f:
        json.dump(res, f)





