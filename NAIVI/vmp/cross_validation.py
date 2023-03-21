from __future__ import annotations
import torch
import gc
from . import VMP
from . import VMP_OPTIONS

prefix = "[CVVMP] "


class CVVMP:

    def __init__(
            self,
            n_nodes: int,
            latent_dim: int,
            binary_covariates: torch.Tensor | None,
            continuous_covariates: torch.Tensor | None,
            edges: torch.Tensor | None,
            edge_index_left: torch.Tensor | None,
            edge_index_right: torch.Tensor | None,
            folds: int = 5,
            **model_args
    ):
        self.binary_covariates = binary_covariates
        self.continuous_covariates = continuous_covariates
        if binary_covariates is not None:
            self.binary_fold_id = torch.randint_like(binary_covariates, folds)
        if continuous_covariates is not None:
            self.continuous_fold_id = torch.randint_like(continuous_covariates, folds)
        self.covariate_elbo = 0.
        self.covariate_log_likelihood = 0.
        self.n_nodes = n_nodes
        self.latent_dim = latent_dim
        self.edges = edges
        self.edge_index_left = edge_index_left
        self.edge_index_right = edge_index_right
        self.folds = folds
        self.model_args = model_args

    def fit_fold(self, fold: int, **fit_args):
        print(f"{prefix}Fitting fold {fold + 1}/{self.folds}")
        X_bin, X_bin_missing, X_cts, X_cts_missing = self._prepare_fold(fold)
        model = VMP(
            n_nodes=self.n_nodes,
            latent_dim=self.latent_dim,
            binary_covariates=X_bin,
            continuous_covariates=X_cts,
            edges=self.edges,
            edge_index_left=self.edge_index_left,
            edge_index_right=self.edge_index_right,
            **self.model_args
        )
        model.fit_and_evaluate(**fit_args)
        # free memory
        print(f"{prefix}Allocated memory: {torch.cuda.memory_allocated() / 1e9} GB")
        del model
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        print(f"{prefix}Allocated memory: {torch.cuda.memory_allocated() / 1e9} GB")

        self.covariate_elbo += model.covariate_elbo(X_bin_missing, X_cts_missing)
        self.covariate_log_likelihood += model.covariate_log_likelihood(X_bin_missing, X_cts_missing)

    def _prepare_fold(self, fold):
        X_bin_missing = None
        X_bin = None
        if self.binary_covariates is not None:
            X_bin_missing = torch.where(
                self.binary_fold_id == fold,
                self.binary_covariates,
                torch.full_like(self.binary_covariates, torch.nan)
            )
            X_bin = torch.where(
                self.binary_fold_id == fold,
                torch.full_like(self.binary_covariates, torch.nan),
                self.binary_covariates
            )
        X_cts_missing = None
        X_cts = None
        if self.continuous_covariates is not None:
            X_cts_missing = torch.where(
                self.continuous_fold_id == fold,
                self.continuous_covariates,
                torch.full_like(self.continuous_covariates, torch.nan)
            )
            X_cts = torch.where(
                self.continuous_fold_id == fold,
                torch.full_like(self.continuous_covariates, torch.nan),
                self.continuous_covariates
            )
        return X_bin, X_bin_missing, X_cts, X_cts_missing

    def fit(self, **fit_args):
        for fold in range(self.folds):
            self.fit_fold(fold, **fit_args)
        return self