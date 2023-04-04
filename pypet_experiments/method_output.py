import torch
import numpy as np

T = torch.Tensor
A = np.ndarray


class MethodOutput:

    def __init__(
            self,
            pred_continuous_covariates: T | A | None = None,
            pred_binary_covariates: T | A | None = None,
            pred_edges: T | A | None = None,
            latent_positions: T | A | None = None,
            latent_heterogeneity: T | A | None = None,
            linear_predictor_covariates: T | A | None = None,
            linear_predictor_edges: T | A | None = None,
            weight_covariates: T | A | None = None,
            bias_covariates: T | A | None = None,
            loss_history: list[float] | T | A | None = None,
    ):
        self.pred_continuous_covariates = pred_continuous_covariates
        self.pred_binary_covariates = pred_binary_covariates
        self.pred_edges = pred_edges
        self.latent_positions = latent_positions
        self.latent_heterogeneity = latent_heterogeneity
        self.linear_predictor_covariates = linear_predictor_covariates
        self.linear_predictor_edges = linear_predictor_edges
        self.weight_covariates = weight_covariates
        self.bias_covariates = bias_covariates
        self.loss_history = loss_history
        self._uniformize()

    def _uniformize(self):
        if isinstance(self.pred_continuous_covariates, np.ndarray):
            self.pred_continuous_covariates = torch.from_numpy(self.pred_continuous_covariates)
        if isinstance(self.pred_binary_covariates, np.ndarray):
            self.pred_binary_covariates = torch.from_numpy(self.pred_binary_covariates)
        if isinstance(self.pred_edges, np.ndarray):
            self.pred_edges = torch.from_numpy(self.pred_edges)
        if isinstance(self.latent_positions, np.ndarray):
            self.latent_positions = torch.from_numpy(self.latent_positions)
        if isinstance(self.latent_heterogeneity, np.ndarray):
            self.latent_heterogeneity = torch.from_numpy(self.latent_heterogeneity)
        if isinstance(self.linear_predictor_covariates, np.ndarray):
            self.linear_predictor_covariates = torch.from_numpy(self.linear_predictor_covariates)
        if isinstance(self.linear_predictor_edges, np.ndarray):
            self.linear_predictor_edges = torch.from_numpy(self.linear_predictor_edges)
        if isinstance(self.weight_covariates, np.ndarray):
            self.weight_covariates = torch.from_numpy(self.weight_covariates)
        if isinstance(self.bias_covariates, np.ndarray):
            self.bias_covariates = torch.from_numpy(self.bias_covariates)
        if isinstance(self.loss_history, np.ndarray):
            self.loss_history = torch.from_numpy(self.loss_history)
        if isinstance(self.loss_history, list):
            self.loss_history = torch.tensor(self.loss_history)

    def cuda(self):
        self.pred_continuous_covariates = self.pred_continuous_covariates.cuda()
        self.pred_binary_covariates = self.pred_binary_covariates.cuda()
        self.pred_edges = self.pred_edges.cuda()
        self.latent_positions = self.latent_positions.cuda()
        self.latent_heterogeneity = self.latent_heterogeneity.cuda()
        self.linear_predictor_covariates = self.linear_predictor_covariates.cuda()
        self.linear_predictor_edges = self.linear_predictor_edges.cuda()
        self.weight_covariates = self.weight_covariates.cuda()
        self.bias_covariates = self.bias_covariates.cuda()
        self.loss_history = self.loss_history.cuda()
        return self