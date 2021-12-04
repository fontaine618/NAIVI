import torch
import numpy as np
import pandas as pd
from pytorch_lightning.metrics.functional import auroc, mean_squared_error
# from NAIVI.utils.metrics import invariant_distance, projection_distance, proba_distance
from NAIVI.utils.base import verbose_init
from torch.optim.lr_scheduler import LambdaLR


class NAIVI:

    def __init__(self, model=None):
        self.model = model
        self.denum = 1.
        self.reg = 0.

    def fit_path(self, train, test=None, reg=None, init=None, **kwargs):
        # TODO: this shouldn't work anymore
        out = {}
        for r in reg:
            self.init(**init)
            out_r = self.fit(train, test, reg=r, **kwargs)
            with torch.no_grad():
                coef_norm = self.model.covariate_model.weight.norm("fro", 1)
                which = torch.where(coef_norm > 0., 1, 0)
                non_zero = which.sum().item()
                out_r.append(non_zero)
                out_r.append(which.cpu().numpy())
            out[r] = out_r
        results = {
            n: []
            for n in ["reg", "llk_train", "mse_train", "auroc_train",
               "dist_inv", "dist_proj", "llk_test", "mse_test", "auroc_test",
                      "aic", "bic",
            "nb_non_zero", "non_zero"]
        }
        best_critrion = np.inf
        best_r = None
        best_out = None
        for r, (_, _,
                llk_train, mse_train, auroc_train,
                dist_inv, dist_proj,
                llk_test, mse_test, auroc_test,
                aic, bic,
                nb_non_zero, non_zero) in out.items():
            results["reg"].append(r)
            results["llk_train"].append(llk_train)
            results["mse_train"].append(mse_train)
            results["auroc_train"].append(auroc_train)
            results["dist_inv"].append(dist_inv)
            results["dist_proj"].append(dist_proj)
            results["llk_test"].append(llk_test)
            results["mse_test"].append(mse_test)
            results["auroc_test"].append(auroc_test)
            results["aic"].append(aic)
            results["bic"].append(bic)
            results["nb_non_zero"].append(nb_non_zero)
            results["non_zero"].append(non_zero)
            if bic < best_critrion:
                best_critrion = bic
                best_r = r
                best_out = out[r]
        return results, best_r, best_out

    def fit(self, train, test=None, reg=0.,
            eps=1.e-6, max_iter=100, optimizer="Rprop", lr=0.01, power=0.0,
            verbose=True, return_log=False, true_values=None
            ):
        if true_values is None:
            true_values = dict()
        for k, v in true_values.items():
            true_values[k] = v.cuda()
        self.reg = reg
        self.compute_denum(train)
        optimizer, scheduler = self.prepare_optimizer(lr=lr, power=power, optimizer=optimizer)
        n_char = verbose_init() if verbose else 0
        epoch, out = 0, None
        outcols = [("train", "grad_norm"),
                   ("train", "loss"), ("train", "mse"), ("train", "auc"),
                   ("ic", "aic"), ("ic", "bic")]
        if test is not None:
            outcols += [("test", "loss"), ("test", "mse"), ("test", "auc")]
        if return_log:
            outcols += [("error", k) for k in true_values.keys()]
            logs = pd.DataFrame(columns=pd.MultiIndex.from_tuples(outcols))
            logs.index.name = "iter"
        for epoch in range(max_iter):
            optimizer.zero_grad()
            # batches = torch.split(torch.randperm(len(train)), batch_size)
            # for batch in batches:
            #     self.batch_update(batch, optimizer, train, epoch)
            self.batch_update(None, optimizer, train, epoch)
            if scheduler is not None:
                scheduler.step()
            converged, grad_norms = self.check_convergence(eps)
            out = self.epoch_metrics(test, train, grad_norms, true_values)
            if return_log:
                df = pd.DataFrame(out, index=[0])
                logs = logs.append(df, ignore_index=True)
            if verbose and epoch % 10 == 0:
                form = "{:<4} {:<12.2e} |" + " {:<12.4f}" * 4
                log = form.format(epoch + 1, out[("train", "grad_L2")],
                                  out[("train", "loss")], out[("train", "mse")],
                                  out[("train", "auc")], out[("train", "auc_A")])
                print(log)
            if converged:
                break
        if verbose:
            print("-" * n_char)
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

    def init(self, positions=None, heterogeneity=None, bias=None, weight=None, sig2=None):
        with torch.no_grad():
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
            out[("train", "auc")], out[("train", "auc_A")] = \
                self.evaluate(train)
            # testing metrics
            if test is not None:
                out[("test", "loss")], out[("test", "mse")], \
                out[("test", "auc")], out[("test", "auc_A")] = self.evaluate(test)
            else:
                out[("test", "loss")], out[("test", "mse")], \
                out[("test", "auc")], out[("test", "auc_A")] = 0., 0., 0., 0.
            # estimation error
            with torch.no_grad():
                for k, v in true_values.items():
                    Z = self.latent_positions()
                    ZZt = torch.mm(Z, Z.t())
                    alpha = self.latent_heterogeneity()
                    Theta_A = alpha[i0] + alpha[i1] + torch.sum(Z[i0, :] * Z[i1, :], 1, keepdim=True)
                    B = self.model.covariate_model.weight
                    B0 = self.model.covariate_model.bias
                    if k == "ZZt":
                        value = ((ZZt - v)**2).sum() / (v**2).sum()
                    if k == "Theta_X":
                        Theta_X = B0.reshape((1, -1)) + torch.mm(Z, B.t())
                        value = ((Theta_X - v)**2).mean()
                    if k == "Theta_A":
                        value = ((Theta_A - v)**2).mean()
                    if k == "P":
                        P = torch.sigmoid(Theta_A)
                        value = ((P - v)**2).mean()
                    if k == "BBt":
                        BBt = torch.mm(B, B.t())
                        value = ((BBt - v)**2).sum() / (v**2).sum()
                    if k == "alpha":
                        value = ((alpha - v)**2).mean()
                    out[("error", k)] = value.item()
        #     # model size
        #     nb_non_zero = self.model.covariate_model.weight.abs().gt(0.).sum().item()
        #     # ICs
        #     out[("ic", "aic")] = out[("train", "loss")] * 2. + nb_non_zero
        #     out[("ic", "bic")] = out[("train", "loss")] * 2. + nb_non_zero * np.log(train.N)
        # # form = "{:<4} {:<10.2e} |" + " {:<11.4f}" * 3 + "|" + \
        # #        " {:<8.4f}" * 2 + "|" + " {:<11.4f}" * 3 + "|" + \
        # #        " {:<11.4f}" * 2
        return out

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
        i0 = i0.cuda() if i0 is not None else i0
        i1 = i1.cuda() if i1 is not None else i1
        A = A.cuda() if A is not None else A
        j = j.cuda() if isinstance(j, torch.Tensor) else j
        if X_cts is not None:
            X_cts = X_cts.cuda()
        if X_bin is not None:
            X_bin = X_bin.cuda()
        return A, X_bin, X_cts, i0, i1, j

    @staticmethod
    def prediction_metrics(X_bin=None, X_cts=None, A=None, mean_cts=None, proba_bin=None, proba_adj=None):
        mse = 0.
        auc = 0.
        auc_A = 0.
        if A is not None:
            auc_A = auroc(proba_adj.clamp_(0., 1.).flatten(), A.int().flatten()).item()
        if X_cts is not None:
            which_cts = ~X_cts.isnan()
            mse = mean_squared_error(mean_cts[which_cts], X_cts[which_cts]).item()
        if X_bin is not None:
            which_bin = ~X_bin.isnan()
            if which_bin.sum() > 0.:
                auc = auroc(proba_bin[which_bin].clamp_(0., 1.), X_bin.int()[which_bin]).item()
        return auc, mse, auc_A

    def batch_update(self, batch, optimizer, train, epoch):
        A, X_bin, X_cts, i0, i1, j = self.get_batch(train, batch)
        # reset gradient
        optimizer.zero_grad()
        # objective
        loss, _, _, _ = self.model.loss_and_fitted_values(i0, i1, j, X_cts, X_bin, A)
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
            # llk /= self.denum
            auc, mse, auc_A = self.prediction_metrics(X_bin, X_cts, A, mean_cts, proba_bin, proba_adj)
        return llk.item(), mse, auc, auc_A

    def check_convergence(self, tol):
        with torch.no_grad():
            grad_Linfty = torch.tensor([
                parm.grad.abs().max()
                for parm in self.model.parameters() if parm.grad is not None
            ]).max().item()
            grad_L1 = torch.tensor([
                parm.grad.abs().sum()
                for parm in self.model.parameters() if parm.grad is not None
            ]).sum().item()
            grad_L2 = torch.tensor([
                (parm.grad**2).sum()
                for parm in self.model.parameters() if parm.grad is not None
            ]).sum().sqrt().item()
            converged = False
            if grad_L2 < tol:
                converged = True
            return converged, {"Linfty": grad_Linfty, "L1": grad_L1, "L2": grad_L2}
