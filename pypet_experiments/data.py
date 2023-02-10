import torch
import os
import math
from pypet import ParameterGroup

T = torch.Tensor


class Dataset:

    def __init__(
            self,
            edge_index_left: T | None = None,
            edge_index_right: T | None = None,
            edges: T | None = None,
            binary_covariates: T | None = None,
            continuous_covariates: T | None = None,
            edges_missing: T | None = None,
            binary_covariates_missing: T | None = None,
            continuous_covariates_missing: T | None = None,
            true_values: dict[str, T] | None = None
    ):
        self.edge_index_left = edge_index_left
        self.edge_index_right = edge_index_right
        self.edges = edges
        self.binary_covariates = binary_covariates
        self.continuous_covariates = continuous_covariates
        self.edges_missing = edges_missing
        self.binary_covariates_missing = binary_covariates_missing
        self.continuous_covariates_missing = continuous_covariates_missing
        if true_values is None:
            true_values = dict()
        self._true_values = true_values

    @property
    def training_set(self) -> tuple[T, T, T, T, T]:
        return (
            self.edge_index_left,
            self.edge_index_right,
            self.edges,
            self.binary_covariates,
            self.continuous_covariates
        )

    @property
    def testing_set(self) -> tuple[T, T, T, T, T]:
        return (
            self.edge_index_left,
            self.edge_index_right,
            self.edges_missing,
            self.binary_covariates_missing,
            self.continuous_covariates_missing
        )

    @property
    def true_values(self) -> dict[str, T]:
        return self._true_values

    @property
    def p_cts(self) -> int:
        if self.continuous_covariates is not None:
            return self.continuous_covariates.shape[1]
        return 0

    @property
    def p_bin(self) -> int:
        if self.binary_covariates is not None:
            return self.binary_covariates.shape[1]
        return 0

    @property
    def n_nodes(self) -> int:
        if self.binary_covariates is not None:
            return self.binary_covariates.shape[0]
        if self.continuous_covariates is not None:
            return self.continuous_covariates.shape[0]
        if self.edge_index_left is not None:
            return self.edge_index_left.max().item() + 1

    @classmethod
    def from_parameters(cls, par: ParameterGroup):
        if par.dataset == "synthetic":
            return cls.synthetic(par)
        elif par.dataset == "real":
            return cls.real(par)
        else:
            raise ValueError("Unknown dataset type: " + par.dataset)

    @classmethod
    def synthetic(cls, par: ParameterGroup):
        p = par.p_cts + par.p_bin
        torch.manual_seed(par.seed)
        weights = torch.randn(par.latent_dim, p)
        bias = torch.randn(1, p)
        latent = torch.randn(par.n_nodes, par.latent_dim) * math.sqrt(par.latent_variance) + \
            par.latent_mean
        heterogeneity = torch.randn(par.n_nodes, 1) * math.sqrt(par.heterogeneity_variance) + \
            par.heterogeneity_mean
        theta_X = latent @ weights + bias
        mean_cts, logit_bin = theta_X[:, :par.p_cts], theta_X[:, par.p_cts:]
        X_cts = torch.randn(par.n_nodes, par.p_cts) * math.sqrt(par.cts_noise) + mean_cts
        X_bin = torch.sigmoid(logit_bin)
        X_bin = torch.bernoulli(X_bin)
        M_cts = torch.rand(par.n_nodes, par.p_cts) < par.missing_covariate_rate
        M_bin = torch.rand(par.n_nodes, par.p_bin) < par.missing_covariate_rate
        X_cts_missing = torch.where(~M_cts, torch.full_like(X_cts, torch.nan), X_cts)
        X_bin_missing = torch.where(~M_bin, torch.full_like(X_bin, torch.nan), X_bin)
        X_cts[M_cts] = torch.nan
        X_bin[M_bin] = torch.nan
        i = torch.tril_indices(par.n_nodes, par.n_nodes, offset=-1)
        i0, i1 = i[0, :], i[1, :]
        theta_A = (latent[i0, :] * latent[i1, :]).sum(1, keepdim=True) + \
            heterogeneity[i0] + heterogeneity[i1]
        A = torch.sigmoid(theta_A)
        A = torch.bernoulli(A)
        M_A = torch.rand_like(A) < par.missing_edge_rate
        A_missing = torch.where(~M_A, torch.full_like(A, torch.nan), A)
        A[M_A] = torch.nan
        return cls(
            edge_index_left=i0,
            edge_index_right=i1,
            edges=A,
            binary_covariates=X_bin,
            continuous_covariates=X_cts,
            edges_missing=A_missing,
            binary_covariates_missing=X_bin_missing,
            continuous_covariates_missing=X_cts_missing,
            true_values=dict(
                latent=latent,
                heterogeneity=heterogeneity,
                weights=weights,
                bias=bias,
                mean_cts=mean_cts,
                logit_bin=logit_bin,
                Theta_A=theta_A,
                Theta_X=theta_X,
                P=torch.sigmoid(theta_A),
                X_cts=X_cts,
                X_bin=X_bin,
                X_cts_missing=X_cts_missing,
                X_bin_missing=X_bin_missing,
                A=A,
                A_missing=A_missing,
                cts_noise=torch.full((par.p_cts, ), par.cts_noise)
            )
        )

    @classmethod
    def real(cls, par: ParameterGroup):
        raise NotImplementedError("Real data not yet supported")

    @property
    def edge_density(self) -> float:
        return self.edges[~torch.isnan(self.edges)].mean().item()

    @property
    def missing_edge_density(self) -> float:
        return self.edges_missing[~torch.isnan(self.edges_missing)].mean().item()

    @property
    def covariate_missing_prop(self) -> float:
        X_cts_missing_n = self.continuous_covariates[~torch.isnan(self.continuous_covariates)].numel()
        X_cts_n = self.continuous_covariates.numel()
        X_bin_missing_n = self.binary_covariates[~torch.isnan(self.binary_covariates)].numel()
        X_bin_n = self.binary_covariates.numel()
        denum = X_cts_n + X_bin_n
        if denum == 0:
            return float("nan")
        return (X_cts_missing_n + X_bin_missing_n) / denum




