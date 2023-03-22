import torch
from collections import defaultdict
from torchmetrics.functional import auroc, mean_squared_error


class NetworkSmoothing:

    def __init__(self):
        pass

    def fit_and_evaluate(
            self,
            binary_covariates: torch.Tensor | None,
            continuous_covariates: torch.Tensor | None,
            binary_covariates_missing: torch.Tensor | None,
            continuous_covariates_missing: torch.Tensor | None,
            edges: torch.Tensor | None,
            edge_index_left: torch.Tensor | None,
            edge_index_right: torch.Tensor | None,
            max_iter=100,
            **kwargs
    ) -> defaultdict[str, float]:
        N = self._get_n_nodes(binary_covariates, continuous_covariates, edge_index_left, edge_index_right)
        adj_matrix = self._compute_adj_matrix(N, edge_index_left, edge_index_right, edges)
        n_neighbors = self._compute_n_neighbors(adj_matrix)
        binary_proba, continuous_mean = self.fit(
            binary_covariates=binary_covariates,
            continuous_covariates=continuous_covariates,
            adj_matrix=adj_matrix,
            n_neighbors=n_neighbors,
            max_iter=max_iter,
        )

        metrics = defaultdict(lambda: float("nan"))
        if binary_covariates_missing is not None:
            obs = binary_covariates_missing[~torch.isnan(binary_covariates_missing)].int()
            proba = binary_proba[~torch.isnan(binary_covariates_missing)]
            metrics["X_bin_auroc"] = auroc(proba, obs, "binary").item() if obs.numel() else float("nan")
        if continuous_covariates_missing is not None:
            mean_cts = continuous_mean[~continuous_covariates_missing.isnan()]
            value = continuous_covariates_missing[~continuous_covariates_missing.isnan()]
            metrics["X_cts_mse"] = mean_squared_error(mean_cts, value).item() if value.numel() else float("nan")
        return metrics

    def fit(
            self,
            binary_covariates: torch.Tensor | None,
            continuous_covariates: torch.Tensor | None,
            adj_matrix: torch.Tensor,
            n_neighbors: torch.Tensor,
            max_iter=100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        binary_proba = None
        continuous_mean = None
        if binary_covariates is not None:
            binary_proba = self._fit(binary_covariates, adj_matrix, n_neighbors, max_iter)
        if continuous_covariates is not None:
            continuous_mean = self._fit(continuous_covariates, adj_matrix, n_neighbors, max_iter)
        return binary_proba, continuous_mean

    def _fit(self, covariates, adj_matrix, n_neighbors, max_iter):
        # start by imputing mean
        colmeans = covariates.nanmean(0).unsqueeze(0).repeat(covariates.shape[0], 1)
        fitted = torch.where(covariates.isnan(), colmeans, covariates)
        for epoch in range(max_iter):
            smoothed = torch.matmul(adj_matrix, fitted) / n_neighbors
            # put true values back in
            fitted = torch.where(covariates.isnan(), smoothed, covariates)
        return fitted

    def _compute_n_neighbors(self, A_mat):
        n_neighbors = A_mat.sum(0).reshape((-1, 1))
        n_neighbors = torch.where(n_neighbors == 0., 1., n_neighbors)  # to avoid division by 0
        return n_neighbors

    def _compute_adj_matrix(self, N, edge_index_left, edge_index_right, edges):
        A_mat = torch.zeros((N, N), device=edges.device)
        A_mat.index_put_((edge_index_left, edge_index_right), edges.flatten())
        A_mat.index_put_((edge_index_right, edge_index_left), edges.flatten())
        return A_mat

    def _get_n_nodes(self, binary_covariates, continuous_covariates, edge_index_left, edge_index_right):
        n_nodes = max(edge_index_left.max(), edge_index_right.max()) + 1
        if binary_covariates is not None:
            n_nodes = max(n_nodes, binary_covariates.shape[0])
        if continuous_covariates is not None:
            n_nodes = max(n_nodes, continuous_covariates.shape[0])
        return n_nodes
