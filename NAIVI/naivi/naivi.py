import torch
import numpy as np
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.metrics.functional import mean_squared_error
from NAIVI.utils.metrics import invariant_distance, projection_distance


class NAIVI:

    def __init__(self, model=None):
        self.model = model
        self.denum = 1.
        self.mnar_penalty = 0.

    def fit(self, train, test=None, Z_true=None, mnar_penalty=0.,
            batch_size=100, eps=1.e-6, max_iter=100,
            lr=0.001, weight_decay=0., verbose=True):
        Z_true = Z_true.cuda()
        self.mnar_penalty = mnar_penalty
        # compute scaling factor
        self.compute_denum(train)
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
        )
        # optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=lr,
        #     weight_decay=weight_decay
        # )
        if verbose:
            n_char = self.verbose_init()
        epoch = 0
        out = None
        for epoch in range(max_iter):
            # custom split
            batches = torch.split(torch.randperm(len(train)), batch_size)
            for batch in batches:
                self.batch_update(batch, optimizer, train, epoch)
            converged, max_abs_grad = self.check_convergence(eps)
            llk_train, out, log = self.epoch_metrics(Z_true, epoch, test, train, max_abs_grad)
            if verbose and epoch % 10 == 0:
                print(log)
            if converged:
                break
        if verbose:
            print("-" * n_char)
        return out

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
            # training metrics
            llk_train, mse_train, auroc_train = self.evaluate(train)
            # testing metrics
            if test is not None:
                llk_test, mse_test, auroc_test = self.evaluate(test)
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
        form = "{:<4} {:<10.2e} |" + " {:<10.4f}" * 3 + "|" + " {:<10.4f}" * 2 + "|" + " {:<10.4f}" * 3
        log = form.format(*out)
        return llk_train, out, log

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
        return self.model.encoder.latent_position_encoder.mean

    def latent_heterogeneity(self):
        return self.model.encoder.latent_heterogeneity_encoder.mean

    def latent_distance(self, Z):
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
        form = "{:<4} {:<10} |" + " {:<10}" * 3 + "|" + " {:<10}" * 2 + "|" + " {:<10}" * 3
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
        # take gradient step
        optimizer.step()
        # projected gradient
        self.model.project()

    def evaluate(self, data):
        with torch.no_grad():
            A, X_bin, X_cts, i0, i1, j = self.get_batch(data, None)
            # get fitted values
            llk, mean_cts, proba_bin, proba_adj = self.model.loss_and_fitted_values(i0, i1, j, X_cts, X_bin, A)
            llk /= self.denum
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
