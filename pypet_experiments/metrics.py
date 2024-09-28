import torch
from torchmetrics.functional import auroc, mean_squared_error, f1_score, accuracy
from .data import Dataset
from .method_output import MethodOutput
from collections import defaultdict


def nanmse(pred, target):
    which = ~torch.isnan(target)
    return mean_squared_error(pred[which], target[which]).item() if which.any() else float("nan")


def nanauroc(pred, target):
    which = ~torch.isnan(target)
    return auroc(pred[which], target[which].long(), task="binary").item() if which.any() else float("nan")


def relmse(pred, target):
    denum = mean_squared_error(target, target).item()
    return mean_squared_error(pred, target).item() / denum if denum else float("nan")

def multiclass_auroc(pred, oh_target):
    proba = pred / pred.sum(dim=1, keepdim=True)
    rowwhich = ~torch.isnan(oh_target).any(dim=1)
    target = oh_target.argmax(dim=1)
    return auroc(
        proba[rowwhich, :], target[rowwhich],
        task="multiclass", average="weighted", num_classes=proba.shape[1]
    ).item() if rowwhich.any() else float("nan")


def multiclass_f1_score(pred, oh_target, average="weighted"):
    pred_class = pred.argmax(dim=1)
    target = oh_target.argmax(dim=1)
    which = ~torch.isnan(oh_target).any(dim=1)
    return f1_score(
        pred_class[which], target[which],
        task="multiclass", average=average, num_classes=pred.shape[1]
    ).item() if which.any() else float("nan")


class Metrics:

    def __init__(self, model_output: MethodOutput, dataset: Dataset):
        self.model_output = model_output
        self.dataset = dataset
        self._metrics = {
            "training": defaultdict(lambda: float("nan")),
            "testing": defaultdict(lambda: float("nan")),
            "estimation": defaultdict(lambda: float("nan"))
        }
        self._compute_metrics()

    @property
    def metrics(self):
        return self._metrics

    def _compute_metrics(self):
        with torch.no_grad():
            self._compute_training_metrics()
            self._compute_test_metrics()
            self._compute_estimation_metrics()

    def _compute_training_metrics(self):
        if self.model_output.pred_continuous_covariates is not None and \
                self.dataset.continuous_covariates is not None:
            self._metrics["training"]["mse_continuous"] = nanmse(
                self.model_output.pred_continuous_covariates,
                self.dataset.continuous_covariates
            )
        if self.model_output.pred_binary_covariates is not None and \
                self.dataset.binary_covariates is not None:
            self._metrics["training"]["auroc_binary"] = nanauroc(
                self.model_output.pred_binary_covariates,
                self.dataset.binary_covariates
            )
            if self.dataset.multiclass_range is not None:
                pred_mat = self.dataset.subset_multiclass(self.model_output.pred_binary_covariates)
                pred_mat = pred_mat.nan_to_num()
                self._metrics["training"]["auroc_multiclass"] = multiclass_auroc(
                    pred_mat,
                    self.dataset.multiclass_covariates
                )
                self._metrics["training"]["f1_multiclass_weighted"] = multiclass_f1_score(
                    pred_mat,
                    self.dataset.multiclass_covariates,
                    average="weighted"
                )
                self._metrics["training"]["f1_multiclass_macro"] = multiclass_f1_score(
                    pred_mat,
                    self.dataset.multiclass_covariates,
                    average="macro"
                )
                self._metrics["training"]["f1_multiclass_micro"] = multiclass_f1_score(
                    pred_mat,
                    self.dataset.multiclass_covariates,
                    average="micro"
                )
        if self.model_output.pred_edges is not None and self.dataset.edges is not None:
            self._metrics["training"]["auroc_edges"] = nanauroc(
                self.model_output.pred_edges,
                self.dataset.edges
            )

    def _compute_test_metrics(self):
        if self.model_output.pred_continuous_covariates is not None and \
                self.dataset.continuous_covariates_missing is not None:
            self._metrics["testing"]["mse_continuous"] = nanmse(
                self.model_output.pred_continuous_covariates,
                self.dataset.continuous_covariates_missing
            )
        if self.model_output.pred_binary_covariates is not None and \
                self.dataset.binary_covariates_missing is not None:
            self._metrics["testing"]["auroc_binary"] = nanauroc(
                self.model_output.pred_binary_covariates,
                self.dataset.binary_covariates_missing
            )
            if self.dataset.multiclass_range is not None:
                pred_mat = self.dataset.subset_multiclass(self.model_output.pred_binary_covariates)
                pred_mat = pred_mat.nan_to_num()
                self._metrics["testing"]["auroc_multiclass"] = multiclass_auroc(
                    pred_mat,
                    self.dataset.multiclass_covariates_missing
                )
                self._metrics["testing"]["f1_multiclass_weighted"] = multiclass_f1_score(
                    pred_mat,
                    self.dataset.multiclass_covariates_missing,
                    average="weighted"
                )
                self._metrics["testing"]["f1_multiclass_macro"] = multiclass_f1_score(
                    pred_mat,
                    self.dataset.multiclass_covariates_missing,
                    average="macro"
                )
                self._metrics["testing"]["f1_multiclass_micro"] = multiclass_f1_score(
                    pred_mat,
                    self.dataset.multiclass_covariates_missing,
                    average="micro"
                )
                labels = self.dataset.multiclass_covariates_missing.argmax(dim=1)
                missing = self.dataset.multiclass_covariates_missing.isnan().any(dim=1)
                self._metrics["testing"]["accuracy_multiclass"] = accuracy(
                    pred_mat.argmax(dim=1)[~missing],
                    labels[~missing],
                    task="multiclass",
                    num_classes=pred_mat.shape[1]
                ).item()
        if self.model_output.pred_edges is not None and self.dataset.edges_missing is not None:
            self._metrics["testing"]["auroc_edges"] = nanauroc(
                self.model_output.pred_edges,
                self.dataset.edges_missing
            )

    def _compute_estimation_metrics(self):
        if self.model_output.latent_heterogeneity is not None and \
            "heterogeneity" in self.dataset.true_values:
            self._metrics["estimation"]["mse_heterogeneity"] = mean_squared_error(
                self.model_output.latent_heterogeneity,
                self.dataset.true_values["heterogeneity"]
            ).item()
            self._metrics["estimation"]["relmse_heterogeneity"] = relmse(
                self.model_output.latent_heterogeneity,
                self.dataset.true_values["heterogeneity"]
            )
        if self.model_output.latent_positions is not None and \
            "latent" in self.dataset.true_values:
            Z_fit = self.model_output.latent_positions
            Z_true = self.dataset.true_values["latent"]
            ZZt_fit = Z_fit @ Z_fit.T
            ZZt_true = Z_true @ Z_true.T
            ZtZ_fit = Z_fit.T @ Z_fit
            ZtZ_true = Z_true.T @ Z_true
            Proj_fit = Z_fit @ torch.inverse(ZtZ_fit) @ Z_fit.T
            Proj_true = Z_true @ torch.inverse(ZtZ_true) @ Z_true.T
            self._metrics["estimation"]["mse_ZZt"] = mean_squared_error(ZZt_fit, ZZt_true).item()
            self._metrics["estimation"]["mse_Proj"] = mean_squared_error(Proj_fit, Proj_true).item()
            self._metrics["estimation"]["relmse_ZZt"] = relmse(ZZt_fit, ZZt_true)
            self._metrics["estimation"]["relmse_Proj"] = relmse(Proj_fit, Proj_true)
        if self.model_output.bias_covariates is not None and \
            "bias" in self.dataset.true_values:
            self._metrics["estimation"]["mse_bias"] = mean_squared_error(
                self.model_output.bias_covariates,
                self.dataset.true_values["bias"]
            ).item()
            self._metrics["estimation"]["relmse_bias"] = relmse(
                self.model_output.bias_covariates,
                self.dataset.true_values["bias"]
            )
        if self.model_output.weight_covariates is not None and \
            "weight" in self.dataset.true_values:
            B_fit = self.model_output.weight_covariates
            B_true = self.dataset.true_values["weight"]
            BBt_fit = B_fit @ B_fit.T
            BBt_true = B_true @ B_true.T
            Proj_fit = B_fit @ torch.inverse(BBt_fit) @ B_fit.T
            Proj_true = B_true @ torch.inverse(BBt_true) @ B_true.T
            self._metrics["estimation"]["mse_BBt"] = mean_squared_error(BBt_fit, BBt_true).item()
            self._metrics["estimation"]["mse_Proj"] = mean_squared_error(Proj_fit, Proj_true).item()
            self._metrics["estimation"]["relmse_BBt"] = relmse(BBt_fit, BBt_true)
            self._metrics["estimation"]["relmse_Proj"] = relmse(Proj_fit, Proj_true)
        if self.model_output.linear_predictor_covariates is not None and \
            "Theta_X" in self.dataset.true_values:
            self._metrics["estimation"]["mse_Theta_X"] = mean_squared_error(
                self.model_output.linear_predictor_covariates,
                self.dataset.true_values["Theta_X"]
            ).item()
            self._metrics["estimation"]["relmse_Theta_X"] = relmse(
                self.model_output.linear_predictor_covariates,
                self.dataset.true_values["Theta_X"]
            )
        if self.model_output.linear_predictor_edges is not None and \
            "Theta_A" in self.dataset.true_values:
            self._metrics["estimation"]["mse_Theta_A"] = mean_squared_error(
                self.model_output.linear_predictor_edges,
                self.dataset.true_values["Theta_A"]
            ).item()
            self._metrics["estimation"]["relmse_Theta_A"] = relmse(
                self.model_output.linear_predictor_edges,
                self.dataset.true_values["Theta_A"]
            )
        if self.model_output.pred_edges is not None and \
            "P" in self.dataset.true_values:
            self._metrics["estimation"]["mse_P"] = mean_squared_error(
                self.model_output.pred_edges,
                self.dataset.true_values["P"]
            ).item()



