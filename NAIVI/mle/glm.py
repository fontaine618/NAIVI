import torch
import copy
from NAIVI.mle.decoder import CovariateModel
from NAIVI.gradient_based.gradient_based import GradientBased


class GLM(GradientBased):

    def __init__(self, K, N, p_cts, p_bin, mnar=False, latent_positions=None):
        super().__init__()
        super().__init__()
        self.mnar = mnar
        self.p_cts = p_cts
        self.p_bin_og = p_bin
        if mnar:
            p_bin += p_bin + p_cts
        self.p_bin = p_bin
        self.model = CovariateModel(K, p_cts, p_bin, N)
        if torch.cuda.is_available():
            self.model.cuda()
            if latent_positions is not None:
                self.positions = latent_positions.cuda()
        else:
            if latent_positions is not None:
                self.positions = latent_positions

    def init(self, positions=None, heterogeneity=None, bias=None, weight=None):
        with torch.no_grad():
            if bias is not None:
                if torch.cuda.is_available():
                    self.model.covariate_model.mean_model.bias.data = bias.view(-1).cuda()
                else:
                    self.model.covariate_model.mean_model.bias.data = bias.view(-1)
            if weight is not None:
                if torch.cuda.is_available():
                    self.model.covariate_model.mean_model.weight.data = weight.t().cuda()
                else:
                    self.model.covariate_model.mean_model.weight.data = weight.t()

    def compute_penalty(self):
        with torch.no_grad():
            coef_norm = self.model.weight.norm("fro", 1)
            if self.reg > 0.:
                coef_norm[:coef_norm.shape[0]//2] = 0.
            return coef_norm.sum().item()

    def compute_denum(self, data):
        _, _, _, _, X_cts, X_bin = data[:]
        if X_cts is not None:
            self.model.n_cts = (~X_cts.isnan()).sum()
        if X_bin is not None:
            self.model.n_bin = (~X_bin.isnan()).sum()
        self.denum = self.model.n_cts + \
                     self.model.n_bin

    def get_batch(self, data, batch=None):
        # get batch
        if batch is None:
            _, _, _, _, X_cts, X_bin = data[:]
            latent_positions = self.positions
        else:
            _, _, _, _, X_cts, X_bin = data[batch]
            latent_positions = self.positions[batch, :]
        if X_cts is not None:
            if torch.cuda.is_available():
                X_cts = X_cts.cuda()
            else:
                X_cts = X_cts
        if X_bin is not None:
            if torch.cuda.is_available():
                X_bin = X_bin.cuda()
            else:
                X_bin = X_bin
        return X_bin, X_cts, latent_positions

    def batch_update(self, batch, optimizer, train, epoch):
        X_bin, X_cts, latent_positions = self.get_batch(train, batch)
        # reset gradient
        optimizer.zero_grad()
        # objective
        loss, _, _ = self.model.loss_and_fitted_values(latent_positions, X_cts, X_bin)
        loss /= self.denum
        # minimum norm solution: add a small ridge penalty
        loss += self.covariate_weight.pow(2.).sum() * 0.01
        # compute gradients
        loss.backward()
        # store previous regression coefficient
        self.store_coef()
        # take gradient step
        optimizer.step()
        # proximal step for regression coefficient
        self.proximal_step(optimizer.param_groups[0]["lr"])

    @property
    def covariate_weight(self):
        return self.model.weight

    def evaluate(self, data):
        with torch.no_grad():
            X_bin, X_cts, latent_positions = self.get_batch(data, None)
            # get fitted values
            llk, mean_cts, proba_bin = self.model.loss_and_fitted_values(latent_positions, X_cts, X_bin)
            llk /= self.denum
            auc, auc_mc, mse, _ = self.prediction_metrics(X_bin, X_cts, None, mean_cts, proba_bin, None)
        return llk.item(), mse, auc, auc_mc, 0.

    def compute_Theta_A(self, i0, i1):
        pass