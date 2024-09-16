from __future__ import annotations
import itertools
import torch
import numpy as np
import math
from typing import Tuple, Dict, Optional
from torchmetrics.functional import auroc, mean_squared_error
from collections import defaultdict
import warnings

from . import MCMC_OPTIONS
import pyro
import pyro.distributions as dist

prefix = "[MCMC] "
T = torch.Tensor


def select_ijx_from_bool(X: T, B: T) -> Tuple[T, T, T]:
    """
    Selects the indices i, j and the values x from a boolean matrix X
    :param X: boolean matrix
    :param B: boolean mask
    :return: i, j, x
    """
    i, j = torch.where(B)
    x = X[i, j]
    return i, j, x


class MCMC:
    """
    NAIVI using HMC in pyro
    """

    _default_hyperparameters = {
        "latent_prior_variance": 1.,
        "latent_prior_mean": 0.,
        "heterogeneity_prior_variance": 1.,
        "heterogeneity_prior_mean": 0.,
        "loading_prior_variance": 100.,
        "cts_variance": (1., 1.)
    }

    def __init__(
        self,
        n_nodes: int,
        latent_dim: int,
        binary_covariates: T | None,
        continuous_covariates: T | None,
        edges: T | None,
        edge_index_left: T | None,
        edge_index_right: T | None,
        **kwargs,
    ):
        self._prepare_hyperparameters(**kwargs)
        self._model: callable = None
        self._dimensions = {
            "n_nodes": n_nodes,
            "latent_dim": latent_dim,
            "p_bin": binary_covariates.size(1) if binary_covariates is not None else 0,
            "p_cts": continuous_covariates.size(1) if continuous_covariates is not None else 0
        }
        self._data = {
            "binary_covariates": binary_covariates,
            "continuous_covariates": continuous_covariates,
            "edges": edges,
            "edge_index_left": edge_index_left,
            "edge_index_right": edge_index_right,
        }
        self._prepare_data()
        self._create_model()

    def _prepare_data(self):
        # binary covariates
        if self._dimensions["p_bin"] > 0:
            self._data["i_bin"], self._data["j_bin"], self._data["x_bin"] = select_ijx_from_bool(
                self._data["binary_covariates"],
                ~torch.isnan(self._data["binary_covariates"])
            )
            self._data["i_bin_missing"], self._data["j_bin_missing"], _ = select_ijx_from_bool(
                self._data["binary_covariates"],
                torch.isnan(self._data["binary_covariates"])
            )
        else:
            self._data["i_bin"], self._data["j_bin"], self._data["x_bin"] = None, None, None
            self._data["i_bin_missing"], self._data["j_bin_missing"], _ = None, None, None
        # continuous covariates
        if self._dimensions["p_cts"] > 0:
            self._data["i_cts"], self._data["j_cts"], self._data["x_cts"] = select_ijx_from_bool(
                self._data["continuous_covariates"],
                ~torch.isnan(self._data["continuous_covariates"])
            )
            self._data["i_cts_missing"], self._data["j_cts_missing"], _ = select_ijx_from_bool(
                self._data["continuous_covariates"],
                torch.isnan(self._data["continuous_covariates"])
            )
        else:
            self._data["i_cts"], self._data["j_cts"], self._data["x_cts"] = None, None, None
            self._data["i_cts_missing"], self._data["j_cts_missing"], _ = None, None, None


    def _prepare_hyperparameters(self, **kwargs):
        self.hyperparameters = self._default_hyperparameters.copy()
        self.hyperparameters.update(kwargs)

    def _create_model(self):
        K = self._dimensions["latent_dim"]
        p_cts = self._dimensions["p_cts"]
        p_bin = self._dimensions["p_bin"]
        n_nodes = self._dimensions["n_nodes"]
        H = self.hyperparameters
        def model(
                i_cts, j_cts, x_cts,
                i_bin, j_bin, x_bin,
                u_edge, v_edge, edge
        ):
            # weights and biases
            if p_cts + p_bin > 0:
                with pyro.plate("j", p_cts + p_bin):
                    bias = pyro.sample(
                        "b0",
                        dist.Normal(0., math.sqrt(H["loading_prior_variance"]))
                    )
                    weights = pyro.sample(
                        "b",
                        dist.MultivariateNormal(
                            torch.zeros(K),
                            H["loading_prior_variance"] * torch.eye(K)
                        )
                    )
            if p_cts > 0:
                with pyro.plate("j_cts", p_cts):
                    variance = 1. / pyro.sample(
                        "variance",
                        dist.Gamma(*H["cts_variance"])
                    )
            # latent variables
            with pyro.plate("u", n_nodes):
                z = pyro.sample(
                    "z",
                    dist.MultivariateNormal(
                        H["latent_prior_mean"] * torch.ones(K),
                        H["latent_prior_variance"] * torch.eye(K)
                    )
                )
                a = pyro.sample(
                    "alpha",
                    dist.Normal(
                        H["latent_prior_mean"],
                        math.sqrt(H["latent_prior_variance"])
                    )
                )
            # continuous attributes
            if p_cts > 0:
                with pyro.plate("cts_ij", i_cts.size(0)):
                    pyro.sample(
                        "x_cts",
                        dist.Normal(
                            bias[j_cts] + (weights[j_cts] * z[i_cts]).sum(-1),
                            variance[j_cts].sqrt()
                        ),
                        obs=x_cts
                    )
            # binary attributes
            if p_bin > 0:
                with pyro.plate("bin_ij", i_bin.size(0)):
                    pyro.sample(
                        "x_bin",
                        dist.Bernoulli(logits=bias[j_bin] + (weights[j_bin + p_cts] * z[i_bin]).sum(-1)),
                        obs=x_bin
                    )
            # edges
            with pyro.plate("uv", len(edge)):
                pyro.sample(
                    "a",
                    dist.Bernoulli(logits=a[u_edge] + a[v_edge] + (
                                z[u_edge, :] * z[v_edge, :]).sum(-1)),
                    obs=edge
                )
        self._model = model
        if MCMC_OPTIONS["logging"]: print(f"{prefix}Model initialization completed")

    def fit(self, num_samples=1000, warmup_steps=200, **kwargs):
        self._run_inference(num_samples=num_samples, warmup_steps=warmup_steps, **kwargs)

    def predict(self):
        if self._mcmc is None:
            raise ValueError("Model not fitted")
        K = self._dimensions["latent_dim"]
        p_cts = self._dimensions["p_cts"]
        p_bin = self._dimensions["p_bin"]
        n_nodes = self._dimensions["n_nodes"]
        samples = self._mcmc.get_samples()
        out = dict()
        if p_cts > 0:
            post_mean = torch.einsum("bjk,bik->bij", samples["b"][:, :p_cts, :], samples["z"])
            post_mean += samples["b0"][:, :p_cts].unsqueeze(1)
            pred_sd = samples["variance"].unsqueeze(1).sqrt().repeat(1, n_nodes, 1)
            post_x = post_mean + torch.randn_like(post_mean) * pred_sd
            out["continuous_attributes"]={
                "samples": post_x,
                "mean": post_x.mean(0),
                "variance": post_x.var(0),
                "mean_variance": post_mean.var(0),
            }
        if p_bin > 0:
            post_logit = torch.einsum("bjk,bik->bij", samples["b"][:, p_cts:, :], samples["z"])
            post_logit += samples["b0"][:, p_cts:].unsqueeze(1)
            post_p = torch.sigmoid(post_logit)
            out["binary_attributes"] = {
                "arithmetic_mean": post_p.mean(0),
                "geometric_mean": post_logit.mean(0).sigmoid(),
                "harmonic_mean": post_p.reciprocal().mean(0).reciprocal()
            }
        post_logit = samples["alpha"].unsqueeze(1) + samples["alpha"].unsqueeze(2) + torch.einsum("bij,bkj->bik", samples["z"], samples["z"])
        post_p = torch.sigmoid(post_logit)
        out["edges"] = {
            "arithmetic_mean": post_p.mean(0),
            "geometric_mean": post_logit.mean(0).sigmoid(),
            "harmonic_mean": post_p.reciprocal().mean(0).reciprocal()
        }
        return out

    def _run_inference(self, num_samples=1000, warmup_steps=200, **kwargs):
        if MCMC_OPTIONS["logging"]: print(f"{prefix}Starting inference")
        nuts = pyro.infer.NUTS(self._model)
        mcmc = pyro.infer.MCMC(nuts, num_samples=num_samples, warmup_steps=warmup_steps, **kwargs)
        mcmc.run(
            self._data["i_cts"], self._data["j_cts"], self._data["x_cts"],
            self._data["i_bin"], self._data["j_bin"], self._data["x_bin"],
            self._data["edge_index_left"], self._data["edge_index_right"], self._data["edges"]
        )
        self._mcmc = mcmc
        if MCMC_OPTIONS["logging"]: print(f"{prefix}Inference completed")

    def get_samples(self):
        if self._mcmc is None:
            raise ValueError("Model not fitted")
        return self._mcmc.get_samples()

    def get_samples_with_derived_quantities(self):
        if self._mcmc is None:
            raise ValueError("Model not fitted")
        samples = self._mcmc.get_samples()
        self._compute_derived_quantities(samples)
        return samples


    def _compute_derived_quantities(self, samples):
        K = self._dimensions["latent_dim"]
        p_cts = self._dimensions["p_cts"]
        p_bin = self._dimensions["p_bin"]
        n_nodes = self._dimensions["n_nodes"]
        H = self.hyperparameters
        # dict_keys(['alpha', 'b', 'b0', 'variance', 'z'])
        # linear predictors for attributes
        if p_cts + p_bin > 0:
            samples["thetaX"] = (
                    samples["b0"].unsqueeze(1) +
                    torch.einsum("bjk,bik->bij", samples["b"], samples["z"])
            )
        # linear predictors for edges
        samples["ZZt"] = torch.einsum("bij,bkj->bik", samples["z"], samples["z"])
        samples["thetaA"] = (
                samples["alpha"].unsqueeze(1) +
                samples["alpha"].unsqueeze(2) +
                samples["ZZt"]
        )
        samples["probA"] = torch.sigmoid(samples["thetaA"])
        # likelihood values
        llks = []
        if p_cts > 0:
            samples["llk_cts"] = dist.Normal(
                samples["thetaX"][:, :, :p_cts],
                samples["variance"].unsqueeze(1).sqrt(),
                validate_args=False # need this for nans
            ).log_prob(self._data["continuous_covariates"].unsqueeze(0))
            llks.append(samples["llk_cts"].nansum((-1, -2)))
        if p_bin > 0:
            samples["llk_bin"] = dist.Bernoulli(
                logits=samples["thetaX"][:, :, p_cts:],
                validate_args=False # need this for nans
            ).log_prob(self._data["binary_covariates"].unsqueeze(0))
            llks.append(samples["llk_bin"].nansum((-1, -2)))
        Amat = torch.zeros_like(samples["thetaA"][0,:,:]).float()
        Amat[self._data["edge_index_left"], self._data["edge_index_right"]] = self._data["edges"]
        Amat.fill_diagonal_(torch.nan)
        samples["llk_edges"] = dist.Bernoulli(
            logits=samples["thetaA"],
            validate_args=False # need this for nans
        ).log_prob(Amat.unsqueeze(0))
        llks.append(samples["llk_edges"].nansum((-1, -2)))
        samples["llk"] = torch.stack(llks, dim=-1).sum(-1)
        # prior values
        lpriors = []
        samples["lprior_z"] = dist.MultivariateNormal(
            H["latent_prior_mean"] * torch.ones(K),
            H["latent_prior_variance"] * torch.eye(K)
        ).log_prob(samples["z"])
        lpriors.append(samples["lprior_z"].sum(-1))
        samples["lprior_alpha"] = dist.Normal(
            H["latent_prior_mean"],
            math.sqrt(H["latent_prior_variance"])
        ).log_prob(samples["alpha"])
        lpriors.append(samples["lprior_alpha"].sum(-1))
        if p_cts + p_bin > 0:
            samples["lprior_b0"] = dist.Normal(0., math.sqrt(H["loading_prior_variance"])).log_prob(samples["b0"])
            samples["lprior_b"] = dist.MultivariateNormal(
                torch.zeros(K),
                H["loading_prior_variance"] * torch.eye(K)
            ).log_prob(samples["b"])
            lpriors.append(samples["lprior_b0"].sum(-1))
        if p_cts > 0:
            samples["lprior_variance"] = dist.Gamma(*H["cts_variance"]).log_prob(1. / samples["variance"])
            lpriors.append(samples["lprior_variance"].sum(-1))
        # joint prior
        samples["lprior"] = torch.stack(lpriors, dim=-1).sum(-1)
        # joint posterior
        samples["lpost"] = samples["llk"] + samples["lprior"]
        # return
        return samples

    def output(self):
        p_cts = self._dimensions["p_cts"]
        p_bin = self._dimensions["p_bin"]
        n_nodes = self._dimensions["n_nodes"]
        samples = self.get_samples_with_derived_quantities()
        pred = self.predict()
        if p_cts > 0:
            pred_continuous_covariates = samples["thetaX"][:, :, :p_cts].mean(0)
            linear_predictor_covariates_cts = samples["thetaX"][:, :, :p_cts].mean(0)
        else:
            pred_continuous_covariates = torch.zeros((n_nodes, 0))
            linear_predictor_covariates_cts = torch.zeros((n_nodes, 0))
        if p_bin > 0:
            pred_binary_covariates = pred["binary_attributes"]["arithmetic_mean"]
            linear_predictor_covariates_bin = samples["thetaX"][:, :, p_cts:].mean(0)
        else:
            pred_binary_covariates = torch.zeros((n_nodes, 0))
            linear_predictor_covariates_bin = torch.zeros((n_nodes, 0))
        linear_predictor_covariates = torch.cat([linear_predictor_covariates_cts, linear_predictor_covariates_bin], dim=1)
        pred_edges = pred["edges"]["arithmetic_mean"]
        linear_predictor_edges = samples["thetaA"].mean(0)
        linear_predictor_edges = linear_predictor_edges[self._data["edge_index_left"], self._data["edge_index_right"]]
        latent_positions = samples["z"].mean(0)
        latent_heterogeneity = samples["alpha"].mean(0)
        if p_cts + p_bin > 0:
            weight_covariates = samples["b"].mean(0)
            bias_covariates = samples["b0"].mean(0)
        else:
            weight_covariates = None
            bias_covariates = None

        return dict(
            pred_continuous_covariates=pred_continuous_covariates,
            pred_binary_covariates=pred_binary_covariates,
            pred_edges=pred_edges,
            latent_positions=latent_positions,
            latent_heterogeneity=latent_heterogeneity,
            linear_predictor_covariates=linear_predictor_covariates,
            linear_predictor_edges=linear_predictor_edges,
            weight_covariates=weight_covariates,
            bias_covariates=bias_covariates,
        )

    def output_with_uncertainty(self):
        p_cts = self._dimensions["p_cts"]
        p_bin = self._dimensions["p_bin"]
        n_nodes = self._dimensions["n_nodes"]
        samples = self.get_samples_with_derived_quantities()
        pred = self.predict()
        if p_cts > 0:
            pred_continuous_covariates = (
                pred["continuous_attributes"]["mean"],
                pred["continuous_attributes"]["variance"]
            )
            linear_predictor_covariates_cts = (
                samples["thetaX"][:, :, :p_cts].mean(0),
                samples["thetaX"][:, :, :p_cts].var(0)
            )
        else:
            pred_continuous_covariates = torch.zeros((n_nodes, 0)), torch.zeros((n_nodes, 0))
            linear_predictor_covariates_cts = torch.zeros((n_nodes, 0)), torch.zeros((n_nodes, 0))
        if p_bin > 0:
            pred_binary_covariates = pred["binary_attributes"]["arithmetic_mean"]
            pred_binary_covariates = (
                pred["binary_attributes"]["arithmetic_mean"],
                pred["binary_attributes"]["arithmetic_mean"].mul(1 - pred["binary_attributes"]["arithmetic_mean"])
            )
            linear_predictor_covariates_bin = (
                samples["thetaX"][:, :, p_cts:].mean(0),
                samples["thetaX"][:, :, p_cts:].var(0)
            )
        else:
            pred_binary_covariates = torch.zeros((n_nodes, 0)), torch.zeros((n_nodes, 0))
            linear_predictor_covariates_bin = torch.zeros((n_nodes, 0)), torch.zeros((n_nodes, 0))
        linear_predictor_covariates = (
            torch.cat([linear_predictor_covariates_cts[0], linear_predictor_covariates_bin[0]], dim=1),
            torch.cat([linear_predictor_covariates_cts[1], linear_predictor_covariates_bin[1]], dim=1)
        )
        pred_edges = (
            pred["edges"]["arithmetic_mean"],
            pred["edges"]["arithmetic_mean"].mul(1 - pred["edges"]["arithmetic_mean"])
        )
        linear_predictor_edges = samples["thetaA"].mean(0), samples["thetaA"].var(0)
        linear_predictor_edges = (
            linear_predictor_edges[0][self._data["edge_index_left"], self._data["edge_index_right"]],
            linear_predictor_edges[1][self._data["edge_index_left"], self._data["edge_index_right"]]
        )
        latent_positions = samples["z"].mean(0)
        latent_positions_cov = samples["z"] - latent_positions.unsqueeze(0)
        latent_positions_cov = torch.einsum("bij,bik->bijk", latent_positions_cov, latent_positions_cov)
        latent_positions_cov = latent_positions_cov.mean(0)
        latent_positions = latent_positions, latent_positions_cov
        latent_heterogeneity = samples["alpha"].mean(0), samples["alpha"].var(0)
        if p_cts + p_bin > 0:
            weight_covariates = samples["b"].mean(0)
            weight_covariates_cov = samples["b"] - weight_covariates.unsqueeze(0)
            weight_covariates_cov = torch.einsum("bij,bik->bijk", weight_covariates_cov, weight_covariates_cov)
            weight_covariates_cov = weight_covariates_cov.mean(0)
            weight_covariates = weight_covariates, weight_covariates_cov
            bias_covariates = samples["b0"].mean(0), samples["b0"].var(0)
        else:
            weight_covariates = None, None
            bias_covariates = None, None

        return dict(
            pred_continuous_covariates=pred_continuous_covariates,
            pred_binary_covariates=pred_binary_covariates,
            pred_edges=pred_edges,
            latent_positions=latent_positions,
            latent_heterogeneity=latent_heterogeneity,
            linear_predictor_covariates=linear_predictor_covariates,
            linear_predictor_edges=linear_predictor_edges,
            weight_covariates=weight_covariates,
            bias_covariates=bias_covariates,
        )




