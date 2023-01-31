from abc import ABC, abstractmethod
from collections import defaultdict
from typing import NamedTuple

import pandas as pd
import torch
import torchmetrics.functional as tmf


class Metric(NamedTuple):
    transform: str | None
    metric: str
    relative: bool

    def __repr__(self):
        out = self.metric
        if self.transform:
            out += f"_{self.transform}"
        if self.relative:
            out += "_rel"
        return out


class Method(ABC):

    """List of all metrics to be evaluated.

    target: [(transformation, metric, relative)]
    """
    _metrics = {
        # estimation
        "latent": [
            Metric("ip", "mse", False),
            Metric("ip", "mse", True),
            Metric("proj", "mse", False),
            Metric("proj", "mse", False),
        ],
        "heterogeneity": [
            Metric(None, "mse", False),
            Metric(None, "mse", True),
        ],
        "linear_predictor_covariates": [
            Metric(None, "mse", False),
            Metric(None, "mse", True),
        ],
        "linear_predictor_edges": [
            Metric(None, "mse", False),
            Metric(None, "mse", True),
        ],
        "edge_probability": [
            Metric(None, "mse", False),
            Metric(None, "mse", True),
        ],
        "weights": [
            Metric("ip", "mse", False),
            Metric("ip", "mse", True),
            Metric("proj", "mse", False),
            Metric("proj", "mse", False),
        ],
        # training
        "binary_covariates": [
            Metric(None, "auroc", False),
        ],
        "continuous_covariates": [
            Metric(None, "mse", False),
        ],
        "edges": [
            Metric(None, "auroc", False),
        ],
        # imputation
        "binary_covariates_missing": [
            Metric(None, "auroc", False),
        ],
        "continuous_covariates_missing": [
            Metric(None, "mse", False),
        ],
    }

    def __init__(self, **kwargs):
        self.metrics_history = defaultdict(lambda: list())

    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def fit_and_evaluate(self, **kwargs):
        pass

    @abstractmethod
    @property
    def logs(self) -> pd.DataFrame:
        """A method that returns the logs after fitting."""
        pass

    @abstractmethod
    def get_estimate(self, name: str) -> torch.Tensor:
        """Needs to implement this method for each quantity to be evaluated.
        The returned value is what the input has to be compared to.

        Should return None if the quantity is not estimated by the method.

        Dimensions expected:
        - latent: (n_samples, n_latent)
        - heterogeneity: (n_samples, )
        - linear_predictor_covariates: (n_samples, n_covariates)
        - linear_predictor_edges: (n_edges, )
        - edge_probability: (n_edges, )
        - weights: (n_covariates, n_latent)
        - binary_covariates: (n_samples, n_covariates)
        - continuous_covariates: (n_samples, n_covariates)
        - edges: (n_edges, )
        - binary_covariates_missing: (n_samples, n_covariates)
        - continuous_covariates_missing: (n_samples, n_covariates)
        """
        pass

    def evaluate(
            self,
            true_values: dict[str, torch.Tensor] | None,
            store: bool = True
    ) -> dict[str, torch.Tensor]:
        if true_values is None:
            true_values = {}
        metrics = dict()
        for name, value in true_values.items():
            metrics.update(self._evaluate(name, value))
        if store:
            self._update_metrics_history(metrics)
        return metrics

    def _evaluate(
            self,
            name: str,
            value: torch.Tensor
    ) -> dict[str, float]:
        metrics = {
            repr(metric): 0
            for metric in self._metrics[name]
        }

    def _update_metrics_history(self, metrics: dict[str, float]):
        for k, v in metrics.items():
            self.metrics_history[k].append(v)