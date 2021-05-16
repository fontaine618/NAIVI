import torch
import numpy as np
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.metrics.functional import mean_squared_error
from NAIVI.utils.metrics import invariant_distance, projection_distance
from torch.optim.lr_scheduler import LambdaLR


class NAIVI:

    def __init__(self, model=None):
        self.model = model
        self.denum = 1.
        self.reg = 0.
        # self.cv_fit_ = None

    # def cv_path(self, train, reg=None, n_folds=5, cv_seed=0, **kwargs):
    #     cv_folds = train.cv_folds(n_folds, cv_seed)
    #     cv_fit = {
    #         fold: copy.deepcopy(self).fit_path(*cv_folds[fold], reg=reg, **kwargs)
    #         for fold in range(n_folds)
    #     }
    #     cv_loss = {
    #         r: torch.tensor([cv_fit[i][r][7] for i in range(n_folds)]).mean().item()
    #         for r in reg
    #     }
    #     self.cv_fit_ = cv_fit
    #     return cv_loss

    def fit_path(self, train, test=None, reg=None, init=None, **kwargs):
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
            "nb_non_zero", "non_zero"]
        }
        best_loss = np.inf
        best_r = None
        best_out = None
        for r, (_, _, llk_train, mse_train, auroc_train,
                dist_inv, dist_proj, llk_test, mse_test, auroc_test,
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
            results["nb_non_zero"].append(nb_non_zero)
            results["non_zero"].append(non_zero)
            if llk_train < best_loss:
                best_loss = llk_train
                best_r = r
                best_out = out[r]
        return results, best_r, best_out

    def fit(self, train, test=None, Z_true=None, reg=0.,
            batch_size=100, eps=1.e-6, max_iter=100,
            lr=0.001, weight_decay=0., verbose=True):
        Z_true = Z_true.cuda() if Z_true is not None else None
        self.reg = reg
        self.compute_denum(train)
        optimizer, scheduler = self.prepare_optimizer(lr)
        n_char = self.verbose_init() if verbose else 0
        epoch, out = 0, None
        for epoch in range(max_iter):
            # for step in ["E", "M"]:
            #     optimizer.zero_grad(set_to_none=True)
            #     self.select_params_for_step(step)
            #     for _ in range(5):
            optimizer.zero_grad()
            # batches = torch.split(torch.randperm(len(train)), batch_size)
            # for batch in batches:
            #     self.batch_update(batch, optimizer, train, epoch)
            self.batch_update(None, optimizer, train, epoch)
            scheduler.step()
            converged, max_abs_grad = self.check_convergence(eps)
            llk_train, out, log = self.epoch_metrics(Z_true, epoch, test, train, max_abs_grad)
            if verbose and epoch % 10 == 0:
                print(log)
            if converged:
                break
        if verbose:
            print("-" * n_char)
        return out

    def select_params_for_step(self, step):
        e = (step == "E")
        m = (step == "M")
        for p in self.model.encoder.parameters():
            p.requires_grad = e
        for p in self.model.covariate_model.parameters():
            p.requires_grad = m
        for p in self.model.adjacency_model.parameters():
            p.requires_grad = m

    def prepare_optimizer(self, lr):
        params = [
            {'params': p, "lr": lr}
            for p in self.model.parameters()
        ]
        optimizer = torch.optim.Adagrad(params)
        # optimizer = torch.optim.Adam(params)
        # optimizer = torch.optim.SGD(params)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1. / (1 + epoch) ** 1.0)
        return optimizer, scheduler

    def init(self, positions=None, heterogeneity=None, bias=None, weight=None):
        with torch.no_grad():
            if positions is not None:
                self.model.encoder.latent_position_encoder.init(positions.cuda())
            if heterogeneity is not None:
                self.model.encoder.latent_heterogeneity_encoder.init(heterogeneity.cuda())
            if bias is not None:
                self.model.covariate_model.mean_model.bias.data = bias.view(-1).cuda()
            if weight is not None:
                self.model.covariate_model.mean_model.weight.data = weight.t().cuda()

    def epoch_metrics(self, Z_true, epoch, test, train, max_abs_grad):
        with torch.no_grad():
            # add penalty to loss
            penalty = self.compute_penalty()
            # training metrics
            llk_train, mse_train, auroc_train = self.evaluate(train)
            llk_train += self.reg * penalty
            # testing metrics
            if test is not None:
                llk_test, mse_test, auroc_test = self.evaluate(test)
                llk_test += self.reg * penalty
            else:
                llk_test, mse_test, auroc_test = 0., 0., 0.
            # distance
            if Z_true is not None:
                dist_inv, dist_proj = self.latent_distance(Z_true)
            else:
                dist_inv, dist_proj = 0., 0.
        out = [epoch, max_abs_grad,
               llk_train, mse_train, auroc_train,
               dist_inv, dist_proj,
               llk_test, mse_test, auroc_test]
        form = "{:<4} {:<10.2e} |" + " {:<11.4f}" * 3 + "|" + " {:<8.4f}" * 2 + "|" + " {:<11.4f}" * 3
        log = form.format(*out)
        return llk_train, out, log

    def compute_penalty(self):
        with torch.no_grad():
            coef_norm = self.model.covariate_model.weight.norm("fro", 1)
            if self.reg > 0.:
                coef_norm[:coef_norm.shape[0]//2] = 0.
            return coef_norm.sum().item()

    def compute_denum(self, data):
        i0, i1, A, j, X_cts, X_bin = data[:]
        self.model.adjacency_model.n_links = len(i0)
        if X_cts is not None:
            self.model.covariate_model.n_cts = (~X_cts.isnan()).sum()
        if X_bin is not None:
            self.model.covariate_model.n_bin = (~X_bin.isnan()).sum()
        self.model.adjacency_model.n_links = len(i0)
        self.denum = self.model.covariate_model.n_cts + \
                     self.model.covariate_model.n_bin + \
                     self.model.adjacency_model.n_links

    def latent_positions(self):
        with torch.no_grad():
            return self.model.encoder.latent_position_encoder.mean

    def latent_heterogeneity(self):
        with torch.no_grad():
            return self.model.encoder.latent_heterogeneity_encoder.mean

    def latent_distance(self, Z):
        with torch.no_grad():
            ZZ = self.latent_positions()
            return invariant_distance(Z, ZZ), projection_distance(Z, ZZ)

    @staticmethod
    def get_batch(data, batch=None):
        # get batch
        if batch is None:
            i0, i1, A, j, X_cts, X_bin = data[:]
        else:
            i0, i1, A, j, X_cts, X_bin = data[batch]
        i0 = i0.cuda()
        i1 = i1.cuda()
        A = A.cuda()
        j = j.cuda()
        if X_cts is not None:
            X_cts = X_cts.cuda()
        if X_bin is not None:
            X_bin = X_bin.cuda()
        return A, X_bin, X_cts, i0, i1, j

    @staticmethod
    def verbose_init():
        form = "{:<4} {:<10} |" + " {:<11}" * 3 + "|" + " {:<8}" * 2 + "|" + " {:<11}" * 3
        names = ["iter", "grad norm", "loss", "mse", "auroc", "inv.", "proj.", "loss", "mse", "auroc"]
        groups = ["", "", "Train", "", "", "Distance", "", "Test", "", ""]
        l1 = form.format(*groups)
        l2 = form.format(*names)
        n_char = len(l1)
        print("-" * n_char)
        print(l1)
        print(l2)
        print("-" * n_char)
        return n_char

    @staticmethod
    def prediction_metrics(X_bin, X_cts, mean_cts, proba_bin):
        mse = 0.
        auc = 0.
        if X_cts is not None:
            which_cts = ~X_cts.isnan()
            mse = mean_squared_error(mean_cts[which_cts], X_cts[which_cts]).item()
        if X_bin is not None:
            which_bin = ~X_bin.isnan()
            if which_bin.sum() > 0.:
                auc = auroc(proba_bin[which_bin], X_bin[which_bin]).item()
        return auc, mse

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
                    mult = torch.where(which == 0., 1., mult)
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
            auc, mse = self.prediction_metrics(X_bin, X_cts, mean_cts, proba_bin)
        return llk.item(), mse, auc

    def check_convergence(self, tol):
        with torch.no_grad():
            max_abs_grad = torch.tensor([
                parm.grad.abs().max()
                for parm in self.model.parameters() if parm.grad is not None
            ]).max().item()
            if max_abs_grad < tol:
                return True, max_abs_grad
            else:
                return False, max_abs_grad
