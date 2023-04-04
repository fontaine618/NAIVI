import pandas as pd
import numpy as np
import torch
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
            true_values: dict[str, T] | None = None,
            multiclass_range: tuple[int, int] | None = None
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
        self.multiclass_range = multiclass_range

    @property
    def multiclass_covariates(self) -> T | None:
        return self.subset_multiclass(self.binary_covariates)

    @property
    def multiclass_covariates_missing(self) -> T | None:
        return self.subset_multiclass(self.binary_covariates_missing)

    def subset_multiclass(self, X: T) -> T:
        return X[:, self.multiclass_range[0]:self.multiclass_range[1]]

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
        elif par.dataset == "email":
            return cls.email(par)
        elif par.dataset == "cora":
            return cls.cora(par)
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
                bias=bias.reshape(-1),
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
    def cora(cls, par: ParameterGroup):
        path_edges = par.path + "/cora.cites"
        path_attributes = par.path + "/cora.content"
        edges = pd.read_table(path_edges, header=None, sep="\t")
        attributes = pd.read_table(path_attributes, header=None, sep="\t")
        attributes.set_index(0, inplace=True)
        word_pa = attributes.iloc[:, :-1]
        labels = attributes.iloc[:, -1]
        label_subset = labels.value_counts().index[-4:]
        which = labels.isin(label_subset)
        word_pa = word_pa[which]
        labels = labels[which]
        labels = pd.get_dummies(labels)
        articles = word_pa.index
        # subset edges
        edges = edges[edges[0].isin(articles) & edges[1].isin(articles)]
        # create node index
        node_index = pd.Series(np.arange(len(articles)), index=articles)
        edges = edges.replace(node_index)
        edges = edges.values
        labels.index = node_index
        n_nodes = labels.shape[0]
        i0, i1 = torch.tril_indices(n_nodes, n_nodes, offset=-1)
        A = torch.zeros(n_nodes, n_nodes)
        A[edges[:, 0], edges[:, 1]] = 1
        A[edges[:, 1], edges[:, 0]] = 1
        A = A[i0, i1].reshape(-1, 1)
        X = torch.tensor(word_pa.values, dtype=torch.float)
        keep = X.sum(0).gt(5.)
        X = X[:, keep]
        y = torch.tensor(labels.values, dtype=torch.float)
        torch.manual_seed(par.seed)
        M_A = torch.rand_like(A) < par.missing_edge_rate
        A_missing = torch.where(~M_A, torch.full_like(A, torch.nan), A)
        A[M_A] = torch.nan

        seeds = []
        for i in range(y.shape[1]):
            rows = y[:, i].nonzero().squeeze()
            seeds.append(rows[torch.randint(0, len(rows), (1,))])
        seeds = torch.cat(seeds)
        M_labels = torch.ones(n_nodes, dtype=torch.bool)
        M_labels[seeds] = False

        M_labels_wide = M_labels.unsqueeze(1).repeat(1, y.shape[1])
        y_missing = torch.where(~M_labels_wide, torch.full_like(y, torch.nan), y)
        y[M_labels, :] = torch.nan

        X_bin = torch.cat([X, y], dim=1)
        X_bin_missing = torch.cat([X * torch.nan, y_missing], dim=1)
        X_cts = torch.empty(n_nodes, 0)
        X_cts_missing = torch.empty(n_nodes, 0)
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
                P=torch.sigmoid(A),
                X_bin=X_bin,
                X_cts=X_cts,
                X_bin_missing=X_bin_missing,
                X_cts_missing=X_cts_missing,
                A=A,
                A_missing=A_missing,
            ),
            multiclass_range=(X.shape[1], X.shape[1] + y.shape[1])
        )

    @classmethod
    def email(cls, par: ParameterGroup):
        path_edges = par.path + "/edges.txt"
        path_labels = par.path + "/dpt_labels.txt"
        edges = pd.read_table(path_edges, header=None, sep=" ").values
        labels = pd.read_table(path_labels, header=None, sep=" ").values
        n_nodes = labels[:, 0].max() + 1
        n_nodes = max(n_nodes, edges.max() + 1)
        i0, i1 = torch.tril_indices(n_nodes, n_nodes, offset=-1)
        A = torch.zeros(n_nodes, n_nodes)
        A[edges[:, 0], edges[:, 1]] = 1
        A[edges[:, 1], edges[:, 0]] = 1
        A = A[i0, i1].reshape(-1, 1)
        torch.manual_seed(par.seed)
        M_A = torch.rand_like(A) < par.missing_edge_rate
        A_missing = torch.where(~M_A, torch.full_like(A, torch.nan), A)
        A[M_A] = torch.nan
        p_bin = labels[:, 1].max() + 1
        X_bin = torch.zeros(n_nodes, p_bin)
        X_bin[labels[:, 0], labels[:, 1]] = 1
        M_bin, M_cts = _create_mask_matrices(
            0, p_bin, par.missing_covariate_rate, n_nodes, par.missing_mechanism
        )
        X_bin_missing = torch.where(~M_bin, torch.full_like(X_bin, torch.nan), X_bin)
        X_bin[M_bin] = torch.nan
        X_cts = torch.zeros(n_nodes, 0)
        X_cts_missing = torch.zeros(n_nodes, 0)
        return cls(
            edge_index_left=i0,
            edge_index_right=i1,
            edges=A,
            edges_missing=A_missing,
            binary_covariates=X_bin,
            binary_covariates_missing=X_bin_missing,
            continuous_covariates=X_cts,
            continuous_covariates_missing=X_cts_missing,
            true_values=dict(
                A=A,
                A_missing=A_missing,
                X_bin=X_bin,
                X_bin_missing=X_bin_missing,
                X_cts=X_cts,
                X_cts_missing=X_cts_missing
            ),
            multiclass_range=(0, p_bin)
        )


    @classmethod
    def facebook(cls, par: ParameterGroup):
        from pypet_experiments.facebook import get_data
        center = par.facebook_center
        path = par.path
        i0, i1, A, X_cts, X_bin = get_data(path, center)
        torch.manual_seed(par.seed)
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
            edges=A.float(),
            binary_covariates=X_bin.float(),
            continuous_covariates=X_cts.float(),
            edges_missing=A_missing.float(),
            binary_covariates_missing=X_bin_missing.float(),
            continuous_covariates_missing=X_cts_missing.float(),
            true_values=dict(
                A=A.float(),
                A_missing=A_missing.float(),
                X_cts=X_cts.float(),
                X_bin=X_bin.float(),
                X_cts_missing=X_cts_missing.float(),
                X_bin_missing=X_bin_missing.float(),
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

