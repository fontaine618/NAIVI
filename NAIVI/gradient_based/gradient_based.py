import torch
import numpy as np
import pandas as pd
from torchmetrics.functional import auroc, mean_squared_error
# from NAIVI.utils.metrics import invariant_distance, projection_distance, proba_distance
from NAIVI.utils.base import verbose_init
from torch.optim.lr_scheduler import LambdaLR
from collections import defaultdict


class GradientBased:

    def __init__(self, model=None):
        self.model = model
        self.denum = 1.
        self.reg = 0.
        self.reg_B = 0.

    def fit_path(self, train, test=None, reg=None, **kwargs):
        if reg is None:
            reg = 10 ** np.linspace(-1, 1, 11)
        elif isinstance(reg, int):
            reg = 10 ** np.linspace(-1, 1, reg)
        elif ~isinstance(reg, np.ndarray):
            reg = 10 ** np.linspace(-1, 1, 11)
        reg = np.sort(reg)[::-1]
        out = {}
        for r in reg:
            print(r)
            out[r] = self.fit(train, test, reg=r, **kwargs)
        return out

    def fit(self, train, test=None, reg=0., reg_B=0.,
            eps=1.e-6, max_iter=100, optimizer="Rprop", lr=0.01, power=0.0,
            verbose=True, return_log=False, true_values=None
            ):
        if true_values is None:
            true_values = dict()
        for k, v in true_values.items():
            if torch.cuda.is_available():
                true_values[k] = v.cuda()
            else:
                true_values[k] = v
        self.reg = reg
        self.reg_B = reg_B
        self.compute_denum(train)
        optimizer, scheduler = self.prepare_optimizer(lr=lr, power=power, optimizer=optimizer)
        n_char = verbose_init() if verbose else 0
        epoch, out = 0, None
        outcols = [("train", "grad_Linfty"), ("train", "grad_L2"), ("train", "grad_L1"),
                   ("train", "loss"), ("train", "mse"),
                   ("train", "auc"), ("train", "auc_multiclass"),
                   ("train", "auc_A"),
                   ("ic", "aic"), ("ic", "bic")]
        if test is not None:
            outcols += [
                ("test", "loss"), ("test", "mse"),
                ("test", "auc"), ("test", "auc_multiclass"), ("test", "auc_A")
            ]
        if return_log:
            outcols += [("error", k) for k in true_values.keys()]
            logs = {k: [] for k in outcols}
        for epoch in range(max_iter+1):
            optimizer.zero_grad()
            self.batch_update(None, optimizer, train, epoch)
            if scheduler is not None:
                scheduler.step()
            converged, grad_norms = self.check_convergence(eps)
            out = self.epoch_metrics(test, train, grad_norms, true_values)
            if return_log:
                for k, v in out.items():
                    logs[k].append(v)
            if verbose and epoch % 10 == 0:
                form = "{:<4} {:<12.2e} |" + " {:<12.4f}" * 4
                log = form.format(epoch, out[("train", "grad_L2")],
                                  out[("train", "loss")], out[("train", "mse")],
                                  out[("train", "auc")], out[("train", "auc_A")])
                print(log)
            if converged:
                break
        if verbose:
            print("-" * n_char)
        out = self.epoch_metrics(test, train, grad_norms, true_values)
        i0, i1, _, _, _, _ = train[:]
        self.Theta_A = self.compute_Theta_A(i0, i1)
        if return_log:
            return out, logs
        else:
            return out

    def prepare_optimizer(self, lr, power=1.0, optimizer="Adam"):
        params = [
            {'params': p, "lr": lr}
            for p in self.model.parameters()
        ]
        optimizer = getattr(torch.optim, optimizer)(params)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1. / (1 + epoch) ** power)
        # scheduler = None
        return optimizer, scheduler

    def init(self, positions=None, heterogeneity=None, bias=None, weight=None, sig2=None, components=None):
        with torch.no_grad():

            if torch.cuda.is_available():
                if positions is not None:
                    if "mean" in positions:
                        self.model.encoder.latent_position_encoder.init(positions["mean"].cuda())
                    if "variance" in positions:
                        self.model.encoder.latent_position_encoder.init(positions["variance"].cuda())
                if heterogeneity is not None:
                    if "mean" in heterogeneity:
                        self.model.encoder.latent_heterogeneity_encoder.init(heterogeneity["mean"].cuda())
                    if "variance" in heterogeneity:
                        self.model.encoder.latent_heterogeneity_encoder.init(heterogeneity["variance"].cuda())
                if bias is not None:
                    self.model.covariate_model.mean_model.bias.data = bias.view(-1).cuda()
                if weight is not None:
                    self.model.covariate_model.mean_model.weight.data = weight.t().cuda()
                if sig2 is not None:
                    self.model.covariate_model.set_var(sig2.cuda())
                if components is not None:
                    self.model.adjacency_model.set_components(components.cuda())
            else:
                if positions is not None:
                    if "mean" in positions:
                        self.model.encoder.latent_position_encoder.init(positions["mean"])
                    if "variance" in positions:
                        self.model.encoder.latent_position_encoder.init(positions["variance"])
                if heterogeneity is not None:
                    if "mean" in heterogeneity:
                        self.model.encoder.latent_heterogeneity_encoder.init(heterogeneity["mean"])
                    if "variance" in heterogeneity:
                        self.model.encoder.latent_heterogeneity_encoder.init(heterogeneity["variance"])
                if bias is not None:
                    self.model.covariate_model.mean_model.bias.data = bias.view(-1)
                if weight is not None:
                    self.model.covariate_model.mean_model.weight.data = weight.t()
                if sig2 is not None:
                    self.model.covariate_model.set_var(sig2)
                if components is not None:
                    self.model.adjacency_model.set_components(components)

    def epoch_metrics(self, test, train, grad_norms, true_values=None):
        if true_values is None:
            true_values = dict()
        out = dict()
        with torch.no_grad():
            i0, i1, A, j, X_cts, X_bin = train[:]
            out[("train", "grad_Linfty")] = grad_norms["Linfty"]
            out[("train", "grad_L1")] = grad_norms["L1"]
            out[("train", "grad_L2")] = grad_norms["L2"]
            # training metrics
            out[("train", "loss")], out[("train", "mse")], \
            out[("train", "auc")], out[("train", "auc_multiclass")], out[("train", "auc_A")] = \
                self.evaluate(train)
            # testing metrics
            if test is not None:
                out[("test", "loss")], out[("test", "mse")], \
                out[("test", "auc")],  out[("test", "auc_multiclass")], out[("test", "auc_A")] = self.evaluate(test)
            else:
                out[("test", "loss")], out[("test", "mse")], \
                out[("test", "auc")],  out[("test", "auc_multiclass")], \
                    out[("test", "auc_A")] = 0., 0., 0., 0., 0.
            # estimation error
            with torch.no_grad():
                for k, v in true_values.items():
                    Z = self.latent_positions()
                    ZZt = torch.mm(Z, Z.t())
                    alpha = self.latent_heterogeneity()
                    W = self.model.adjacency_model.components
                    Theta_A = alpha[i0] + alpha[i1] + torch.sum(Z[i0, :] * W * Z[i1, :], 1, keepdim=True)
                    B = self.model.covariate_model.weight
                    B0 = self.model.covariate_model.bias
                    if self.model.mnar:
                        B = B[:(self.model.p_cts+self.model.p_bin)//2, :]
                        B0 = B0[:(self.model.p_cts+self.model.p_bin)//2]
                    if k == "ZZt":
                        value = ((ZZt - v)**2).sum() / (v**2).sum()
                    if k == "Theta_X":
                        Theta_X = B0.reshape((1, -1)) + torch.mm(Z, B.t())
                        value = ((Theta_X - v)**2).sum() / (v**2).sum()
                    if k == "Theta_A":
                        value = ((Theta_A - v)**2).sum() / (v**2).sum()
                    if k == "P":
                        P = torch.sigmoid(Theta_A)
                        value = ((P - v)**2).mean()
                    if k == "BBt":
                        BBt = torch.mm(B, B.t())
                        value = ((BBt - v)**2).sum() / (v**2).sum()
                    if k == "alpha":
                        value = ((alpha - v)**2).mean()
                    out[("error", k)] = value.item()
        return out

    @property
    def Theta_X(self):
        Z = self.latent_positions()
        B = self.model.covariate_model.weight
        B0 = self.model.covariate_model.bias
        if self.model.mnar:
            B = B[:(self.model.p_cts+self.model.p_bin)//2, :]
            B0 = B0[:(self.model.p_cts+self.model.p_bin)//2]
        return B0.reshape((1, -1)) + torch.mm(Z, B.t())

    def compute_Theta_A(self, i0, i1):
        Z = self.latent_positions()
        alpha = self.latent_heterogeneity()
        W = self.model.adjacency_model.components
        return alpha[i0] + alpha[i1] + torch.sum(Z[i0, :] * W * Z[i1, :], 1, keepdim=True)

    def compute_penalty(self):
        with torch.no_grad():
            coef_norm = self.model.covariate_model.weight.norm("fro", 1)
            if self.reg > 0.:
                coef_norm[:coef_norm.shape[0]//2] = 0.
            return coef_norm.sum().item()

    def compute_denum(self, data):
        i0, i1, A, j, X_cts, X_bin = data[:]
        N = data.N
        K = self.model.K
        self.model.adjacency_model.n_links = 1 if i0 is None else len(i0)
        if X_cts is not None:
            self.model.covariate_model.n_cts = (~X_cts.isnan()).sum()
        if X_bin is not None:
            self.model.covariate_model.n_bin = (~X_bin.isnan()).sum()
        self.denum = self.model.covariate_model.n_cts + \
                     self.model.covariate_model.n_bin + \
                     self.model.adjacency_model.n_links * self.model.network_weight + \
                     (N+1) * K

    def latent_positions(self):
        with torch.no_grad():
            return self.model.encoder.latent_position_encoder.mean

    def latent_heterogeneity(self):
        with torch.no_grad():
            return self.model.encoder.latent_heterogeneity_encoder.mean

    @staticmethod
    def get_batch(data, batch=None):
        # get batch
        if batch is None:
            i0, i1, A, j, X_cts, X_bin = data[:]
        else:
            i0, i1, A, j, X_cts, X_bin = data[batch]

        if torch.cuda.is_available():
            i0 = i0.cuda() if i0 is not None else i0
            i1 = i1.cuda() if i1 is not None else i1
            A = A.cuda() if A is not None else A
            j = j.cuda() if isinstance(j, torch.Tensor) else j
            if X_cts is not None:
                X_cts = X_cts.cuda()
            if X_bin is not None:
                X_bin = X_bin.cuda()
        else:
            i0 = i0
            i1 = i1
            A = A
            j = j
            if X_cts is not None:
                X_cts = X_cts
            if X_bin is not None:
                X_bin = X_bin

        return A, X_bin, X_cts, i0, i1, j

    @staticmethod
    def prediction_metrics(X_bin=None, X_cts=None, A=None, mean_cts=None, proba_bin=None, proba_adj=None):
        mse = 0.
        auc = 0.
        auc_mc = 0.
        auc_A = 0.
        if A is not None:
            auc_A = 0.5 #auroc(proba_adj.clamp_(0., 1.).flatten(), A.int().clamp_(0, 1).flatten(), task="binary").item()
        if X_cts is not None:
            which_cts = ~X_cts.isnan()
            mse = mean_squared_error(mean_cts[which_cts], X_cts[which_cts]).item()
        if X_bin is not None:
            which_bin = ~X_bin.isnan()
            if which_bin.sum() > 0.:
                auc = 0.5
                # auroc(
                #     proba_bin[which_bin].clamp_(0., 1.),
                #     X_bin.int()[which_bin],
                #     task="binary"
                # ).item()
            which_rows = which_bin.sum(dim=1) > 0
            if which_rows.sum() > 0.:
                proba_multiclass = proba_bin / proba_bin.sum(dim=1, keepdim=True)
                obs_multiclass = (X_bin == 1.).int().clamp_(0, 1).argmax(dim=1)
                auc_mc = 0.5
                # auroc(
                #     proba_multiclass[which_rows, :],
                #     obs_multiclass[which_rows].int().clamp_(0, 1),
                #     task="multiclass", average="weighted",
                #     num_classes=proba_multiclass.shape[1]
                # ).item()
        return auc, auc_mc, mse, auc_A

    def batch_update(self, batch, optimizer, train, epoch):
        A, X_bin, X_cts, i0, i1, j = self.get_batch(train, batch)
        # reset gradient
        optimizer.zero_grad()
        # objective
        loss, _, _, _ = self.model.loss_and_fitted_values(i0, i1, j, X_cts, X_bin, A)
        # loss += self.reg_B * (self.covariate_weight ** 2).nansum()
        loss /= self.denum
        # compute gradients
        loss.backward()
        # store previous regression coefficient
        self.store_coef()
        # take gradient step
        optimizer.step()
        # proximal step for regression coefficient
        self.proximal_step(optimizer.param_groups[0]["lr"])
        # projected gradient for MLE
        self.model.project()

    @property
    def covariate_weight(self):
        return self.model.covariate_model.weight

    @property
    def covariate_bias(self):
        return self.model.covariate_model.bias

    def proximal_step(self, lr):
        if self.covariate_weight.grad is not None:
            with torch.no_grad():
                reg = self.reg
                coef = self.covariate_weight
                prev_data = coef.prev_data
                grad = coef.grad
                data = prev_data - lr * grad
                if reg > 0.:
                    norm = data.norm("fro", 1)
                    mult = (1. - reg * lr / norm).double()
                    mult = torch.where(mult < 0., 0., mult)
                    which = torch.ones_like(mult)
                    which[:which.shape[0]//2] = 0.
                    mult = torch.where(which.eq(0.), 1., mult)
                    data *= mult.reshape((-1, 1))
                coef.data = data

    def store_coef(self):
        with torch.no_grad():
            self.covariate_weight.prev_data = self.covariate_weight.data

    def evaluate(self, data):
        with torch.no_grad():
            A, X_bin, X_cts, i0, i1, j = self.get_batch(data, None)
            # get fitted values
            llk, mean_cts, proba_bin, proba_adj = self.model.loss_and_fitted_values(i0, i1, j, X_cts, X_bin, A)
            # llk += self.reg_B * (self.covariate_weight ** 2).nansum()
            # llk /= self.denum
            auc, auc_mc, mse, auc_A = self.prediction_metrics(X_bin, X_cts, A, mean_cts, proba_bin, proba_adj)
        return llk.item(), mse, auc, auc_mc, auc_A

    def check_convergence(self, tol):
        with torch.no_grad():
            grad_Linfty = torch.tensor([
                parm.grad.abs().max() if parm.grad.numel() else 0.
                for parm in self.model.parameters() if parm.grad is not None
            ]).max().item()
            grad_L1 = torch.tensor([
                parm.grad.abs().nansum()
                for parm in self.model.parameters() if parm.grad is not None
            ]).nansum().item()
            grad_L2 = torch.tensor([
                (parm.grad**2).nansum()
                for parm in self.model.parameters() if parm.grad is not None
            ]).nansum().sqrt().item()
            converged = False
            if grad_L2 < tol:
                converged = True
            return converged, {"Linfty": grad_Linfty, "L1": grad_L1, "L2": grad_L2}

    def results(self, true_values: dict[str, torch.Tensor] | None = None) -> dict[str, float]:
        if true_values is None:
            true_values = {}
        metrics = defaultdict(lambda: float("nan"))
        for name, value in true_values.items():
            metrics.update(self._evaluate(name, value))
        return metrics

    def output(self, data):
        i0, i1, A, j, X_cts, X_bin = data[:]
        _, mean_cts, proba_bin, proba_adj = self.model.loss_and_fitted_values(
            i0, i1, j, X_cts, X_bin, A
        )
        return dict(
            pred_continuous_covariates=mean_cts,
            pred_binary_covariates=proba_bin,
            pred_edges=proba_adj,
            latent_positions=self.latent_positions(),
            latent_heterogeneity=self.latent_heterogeneity(),
            linear_predictor_covariates=self.Theta_X,
            linear_predictor_edges=self.Theta_A,
            weight_covariates=self.covariate_weight,
            bias_covariates=self.covariate_bias
        )

    def _evaluate(self, name: str, value: torch.Tensor | None) -> dict[str, float]:
        metrics = dict()
        if value is None:
            return metrics
        if name == "heterogeneity":
            post = self.latent_heterogeneity()
            diff = (post - value).abs()
            metrics["heteregeneity_l2"] = diff.norm().item()
            metrics["heteregeneity_l2_rel"] = diff.pow(2.).mean().item()
        elif name == "latent":
            post = self.latent_positions()
            ZZt = post @ post.T
            ZtZinv = torch.linalg.inv(post.T @ post)
            Proj = post @ ZtZinv @ post.T

            ZZt0 = value @ value.T
            ZtZinv0 = torch.linalg.inv(value.T @ value)
            Proj0 = value @ ZtZinv0 @ value.T

            metrics["latent_ZZt_fro"] = (ZZt - ZZt0).norm().item()
            metrics["latent_ZZt_fro_rel"] = (metrics["latent_ZZt_fro"] / ZZt0.norm()).item() ** 2
            metrics["latent_Proj_fro"] = (Proj - Proj0).norm().item()
            metrics["latent_Proj_fro_rel"] = (metrics["latent_Proj_fro"] / Proj0.norm()).item() ** 2
        elif name == "bias":
            bias = self.model.covariate_model.bias
            if bias.shape[-1] == 0:
                return metrics
            diff = (bias - value).abs()
            metrics["bias_l2"] = (diff ** 2).sum().sqrt().item()
            metrics["bias_l2_rel"] = (metrics["bias_l2"] / value.norm()).item() ** 2
        elif name == "weights":
            weights = self.model.covariate_model.weight
            if weights.shape[0] == 0:
                return metrics
            ZZt = weights @ weights.T
            ZtZinv = torch.linalg.inv(weights.T @ weights)
            Proj = weights @ ZtZinv @ weights.T

            value = value.T
            ZZt0 = value @ value.T
            ZtZinv0 = torch.linalg.inv(value.T @ value)
            Proj0 = value @ ZtZinv0 @ value.T

            metrics["weights_BBt_fro"] = (ZZt - ZZt0).norm().item()
            metrics["weights_BBt_fro_rel"] = (metrics["weights_BBt_fro"] / ZZt0.norm()).item() ** 2
            metrics["weights_Proj_fro"] = (Proj - Proj0).norm().item()
            metrics["weights_Proj_fro_rel"] = (metrics["weights_Proj_fro"] / Proj0.norm()).item() ** 2
        elif name == "cts_noise":
            var = self.model.covariate_model.cts_noise
            metrics["cts_noise_l2"] = (var - value).norm().item()
            metrics["cts_noise_sqrt_l2"] = (var.sqrt() - value.sqrt()).norm().item()
            metrics["cts_noise_log_l2"] = (var.log() - value.log()).norm().item()
        elif name == "Theta_X":
            metrics["Theta_X_l2"] = (value - self.Theta_X).norm().item()
            metrics["Theta_X_l2_rel"] = (metrics["Theta_X_l2"] / value.norm()).item() ** 2
        elif name == "Theta_A":
            theta_A = self.Theta_A
            metrics["Theta_A_l2"] = (value - theta_A).norm().item()
            metrics["Theta_A_l2_rel"] = (metrics["Theta_A_l2"] / value.norm()).item() ** 2
        elif name == "P":
            P = torch.sigmoid(self.Theta_A)
            metrics["P_l2"] = (value - P).pow(2.).mean().item()
        else:
            # could print a warning message, but that would appear every iteration ...
            pass
        # update history
        return metrics
