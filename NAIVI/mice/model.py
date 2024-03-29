import torch
from collections import defaultdict
import numpy as np
from torchmetrics.functional import auroc, mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression


class MICE:

    def __init__(
        self,
        binary_covariates: torch.Tensor | None = None,
        continuous_covariates: torch.Tensor | None = None,
        max_iter: int = 100,
        rel_tol: float = 1e-3,
    ):
        p_cts, p_bin, X = self._process_input(binary_covariates, continuous_covariates)
        self.p_cts = p_cts
        self.p_bin = p_bin
        self.X = X
        self.estimator = BayesianRidge()
        Xmin = torch.where(torch.isnan(X), torch.full_like(X, float("inf")), X).min(0)[0]
        Xmax = torch.where(torch.isnan(X), torch.full_like(X, float("-inf")), X).max(0)[0]
        Xmax = torch.where(Xmax == Xmin, Xmin + 1, Xmax)
        self.model = IterativeImputer(
            random_state=0, estimator=self.estimator, imputation_order="random",
            max_iter=max_iter, tol=rel_tol,
            verbose=2, skip_complete=True,
            min_value=Xmin.cpu(),
            max_value=Xmax.cpu()
        )

    def _process_input(self, binary_covariates, continuous_covariates):
        # n_nodes = 0
        p_cts = 0
        p_bin = 0
        X = None
        if continuous_covariates is not None:
            # n_nodes = max(n_nodes, continuous_covariates.shape[0])
            p_cts = continuous_covariates.shape[1]
            if X is None:
                X = torch.zeros((continuous_covariates.shape[0], 0))
            X = torch.cat([X, continuous_covariates], 1)
        if binary_covariates is not None:
            # n_nodes = max(n_nodes, binary_covariates.shape[0])
            p_bin = binary_covariates.shape[1]
            if X is None:
                X = torch.zeros((binary_covariates.shape[0], 0))
            X = torch.cat([X, binary_covariates], 1)
        return p_cts, p_bin, X

    def fit_and_evaluate(
        self,
        binary_covariates: torch.Tensor | None = None,
        continuous_covariates: torch.Tensor | None = None,
    ) -> defaultdict[str, float]:
        X_pred = self.model.fit_transform(self.X.cpu())
        X_cts_pred, X_bin_pred, _ = np.split(
            X_pred, indices_or_sections=[self.p_cts, self.p_cts + self.p_bin], axis=1
        )
        metrics = defaultdict(lambda: float("nan"))
        if binary_covariates is not None:
            obs = binary_covariates[~torch.isnan(binary_covariates)].int()
            proba = torch.Tensor(X_bin_pred)[~torch.isnan(binary_covariates)]
            metrics["X_bin_auroc"] = auroc(proba, obs, "binary").item() if obs.numel() else float("nan")
        if continuous_covariates is not None:
            mean_cts = torch.Tensor(X_cts_pred)[~continuous_covariates.isnan()]
            value = continuous_covariates[~continuous_covariates.isnan()]
            metrics["X_cts_mse"] = mean_squared_error(mean_cts, value).item() if value.numel() else float("nan")
        return metrics

    def fit_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        X_pred = self.model.fit_transform(self.X.cpu())
        X_cts_pred, X_bin_pred, _ = np.split(
            X_pred, indices_or_sections=[self.p_cts, self.p_cts + self.p_bin], axis=1
        )
        return torch.Tensor(X_bin_pred), torch.Tensor(X_cts_pred)


class KNN(MICE):

    def __init__(
        self,
        binary_covariates: torch.Tensor | None = None,
        continuous_covariates: torch.Tensor | None = None,
        n_neighbors: int = 5,
    ):
        super().__init__(binary_covariates, continuous_covariates)
        self.model = KNNImputer(n_neighbors=n_neighbors, weights="uniform", metric="nan_euclidean")