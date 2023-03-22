import torch
from collections import defaultdict
from torchmetrics.functional import auroc, mean_squared_error


class Mean:

    def __init__(
            self,
            binary_covariates: torch.Tensor | None = None,
            continuous_covariates: torch.Tensor | None = None,
    ):
        self.bin_mean = None
        self.cts_mean = None
        if binary_covariates is not None:
            self.bin_mean = torch.nanmean(binary_covariates, dim=0)
        if continuous_covariates is not None:
            self.cts_mean = torch.nanmean(continuous_covariates, dim=0)

    def evaluate(
            self,
            binary_covariates: torch.Tensor | None = None,
            continuous_covariates: torch.Tensor | None = None,
    ) -> defaultdict[str, float]:
        metrics = defaultdict(lambda: float("nan"))
        if binary_covariates is not None:
            proba = self.bin_mean.unsqueeze(0).repeat(binary_covariates.shape[0], 1)
            obs = binary_covariates[~torch.isnan(binary_covariates)].int()
            proba = proba[~torch.isnan(binary_covariates)]
            metrics["X_bin_auroc"] = auroc(proba, obs, "binary").item() if obs.numel() else float("nan")
        if continuous_covariates is not None:
            mean_cts = self.cts_mean.unsqueeze(0).repeat(continuous_covariates.shape[0], 1)
            mean_cts = mean_cts[~continuous_covariates.isnan()]
            value = continuous_covariates[~continuous_covariates.isnan()]
            metrics["X_cts_mse"] = mean_squared_error(mean_cts, value).item() if value.numel() else float("nan")
        return metrics
