import torch
import os
import math
from pypet import ParameterGroup
from NAIVI.utils.data import JointDataset

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
        elif par.dataset == "facebook":
            return cls.facebook(par)
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
        M_bin, M_cts = _create_mask_matrices(
            par.p_cts, par.p_bin, par.missing_covariate_rate, par.n_nodes, par.missing_mechanism
        )
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
                cts_noise=torch.full((par.p_cts, ), par.cts_noise),
                latnet_dim=par.latent_dim,
            )
        )

    @classmethod
    def facebook(cls, par: ParameterGroup):
        from .datasets.facebook import get_data
        center = par.facebook_center
        path = par.path
        i0, i1, A, X_cts, X_bin = get_data(path, center)
        M_A = torch.rand_like(A) < par.missing_edge_rate
        A_missing = torch.where(~M_A, torch.full_like(A, torch.nan), A)
        A[M_A] = torch.nan
        p_cts = X_cts.shape[1] if X_cts is not None else 0
        p_bin = X_bin.shape[1] if X_bin is not None else 0
        n_nodes = max(i0.max().item(), i1.max().item()) + 1
        if X_cts is not None:
            n_nodes = max(n_nodes, X_cts.shape[0])
        else:
            X_cts = torch.empty((n_nodes, 0))
        if X_bin is not None:
            n_nodes = max(n_nodes, X_bin.shape[0])
        else:
            X_bin = torch.empty((n_nodes, 0))
        M_bin, M_cts = _create_mask_matrices(
            p_cts, p_bin, par.missing_covariate_rate, n_nodes, par.missing_mechanism
        )
        X_cts_missing = torch.where(~M_cts, torch.full_like(X_cts, torch.nan), X_cts)
        X_cts[M_cts] = torch.nan
        X_bin_missing = torch.where(~M_bin, torch.full_like(X_bin, torch.nan), X_bin)
        X_bin[M_bin] = torch.nan
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
                A=A,
                A_missing=A_missing,
                X_cts=X_cts,
                X_bin=X_bin,
                X_cts_missing=X_cts_missing,
                X_bin_missing=X_bin_missing,
            )
        )

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

    def to_JointDataset_train(self, cuda: bool = True) -> JointDataset:
        return JointDataset(
            i0=self.edge_index_left,
            i1=self.edge_index_right,
            A=self.edges,
            X_cts=self.continuous_covariates,
            X_bin=self.binary_covariates,
            cuda=cuda,
            return_missingness=False,
            test=False
        )

    def to_JointDataset_test(self, cuda: bool = True) -> JointDataset:
        return JointDataset(
            i0=self.edge_index_left,
            i1=self.edge_index_right,
            A=self.edges_missing,
            X_cts=self.continuous_covariates_missing,
            X_bin=self.binary_covariates_missing,
            cuda=cuda,
            return_missingness=False,
            test=True
        )


def _create_mask_matrices(p_cts, p_bin, missing_covariate_rate, n_nodes, missing_mechanism):
    p = p_cts + p_bin
    if missing_mechanism == "uniform":
        M_cts = torch.rand(n_nodes, p_cts) < missing_covariate_rate
        M_bin = torch.rand(n_nodes, p_bin) < missing_covariate_rate
    elif missing_mechanism == "row_deletion":
        M_cts = torch.rand(n_nodes, 1) < missing_covariate_rate
        M_bin = torch.rand(n_nodes, 1) < missing_covariate_rate
        M_cts = M_cts.repeat(1, p_cts)
        M_bin = M_bin.repeat(1, p_bin)
    elif missing_mechanism == "block":
        sqrt_rate = math.sqrt(missing_covariate_rate)
        rows = torch.rand(n_nodes, 1) < sqrt_rate
        cols_cts = torch.rand(1, p_cts) < sqrt_rate
        cols_bin = torch.rand(1, p_bin) < sqrt_rate
        M_cts = rows.repeat(1, p_cts) & cols_cts.repeat(n_nodes, 1)
        M_bin = rows.repeat(1, p_bin) & cols_bin.repeat(n_nodes, 1)
    elif missing_mechanism == "triangle":
        missing_props = torch.linspace(p / 3, p + p / 3, p)
        missing_props = p * missing_covariate_rate * missing_props / missing_props.sum()
        missing_props = missing_props[torch.randperm(p)].reshape(1, p).repeat(n_nodes, 1)
        M_cts = torch.rand(n_nodes, p_cts) < missing_props[:, :p_cts]
        M_bin = torch.rand(n_nodes, p_bin) < missing_props[:, p_cts:]
    else:
        raise ValueError("Unknown missing mechanism: " + missing_mechanism)
    return M_bin, M_cts

