import torch
import numpy as np
from pytorch_lightning.metrics.functional import auroc
from pytorch_lightning.metrics.functional import mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


class MICE:

    def __init__(self, K, N, p_cts, p_bin):
        estimator = BayesianRidge()
        self.model = IterativeImputer(random_state=0, estimator=estimator,
                                      imputation_order="random", max_iter=10)
        self.p_cts = p_cts
        self.p_bin = p_bin
        self.N = N

    def fit(self, train, test=None, Z_true=None,
            batch_size=100, eps=1.e-6, max_iter=100,
            lr=0.001, weight_decay=0., n_sample=1):
        l = self.verbose_init()
        # get train data
        _, _, _, _, X_cts, X_bin = train[:]
        if X_cts is None:
            X_cts = torch.zeros((self.N, 0))
        if X_bin is None:
            X_bin = torch.zeros((self.N, 0))
        # concatenate
        X = torch.cat([X_cts, X_bin], 1)
        # predict
        X_pred = self.model.fit_transform(X)
        X_cts_pred, X_bin_pred, _ = np.split(X_pred,
                                          indices_or_sections=[self.p_cts, self.p_cts+self.p_bin],
                                          axis=1)
        # get test data
        _, _, _, _, X_test_cts, X_test_bin = test[:]
        if self.p_cts > 0:
            X_cts_pred[np.isnan(X_test_cts) == 1.] = np.nan
            X_cts_pred = torch.tensor(X_cts_pred)
        if self.p_bin > 0:
            X_bin_pred[np.isnan(X_test_bin) == 1.] = np.nan
            X_bin_pred = X_bin_pred.clip(0., 1.)
            X_bin_pred = torch.tensor(X_bin_pred)

        out = self.metrics(X_test_bin, X_test_cts, X_cts_pred, X_bin_pred)
        print("-" * l)
        return out

    def verbose_init(self):
        # verbose
        form = "{:<4} |" + " {:<10}" * 3 + "|" + " {:<10}" * 2 + "|" + " {:<10}" * 3
        names = ["iter", "loss", "mse", "auroc", "inv.", "proj.", "loss", "mse", "auroc"]
        groups = ["", "Train", "", "", "Distance", "", "Test", "", ""]
        l1 = form.format(*groups)
        l2 = form.format(*names)
        l = len(l1)
        print("-" * l)
        print(l1)
        print(l2)
        print("-" * l)
        return l

    def metrics(self, X_bin, X_cts, mean_cts, proba_bin):
        auroc_test, mse_test = self.prediction_metrics(X_bin, X_cts, mean_cts, proba_bin)
        out = [0, 0., 0., 0., 0., 0., 0., mse_test, auroc_test]
        form = "{:<4} |" + " {:<10.4f}" * 3 + "|" + " {:<10.4f}" * 2 + "|" + " {:<10.4f}" * 3
        print(form.format(*out))
        return out

    def init(self, positions=None, heterogeneity=None, bias=None, weight=None):
        return None

    def prediction_metrics(self, X_bin, X_cts, mean_cts, proba_bin):
        mse = 0.
        auc = 0.
        if X_cts is not None:
            which_cts = ~X_cts.isnan()
            mse = mean_squared_error(mean_cts[which_cts], X_cts[which_cts]).item()
        if X_bin is not None:
            which_bin = ~X_bin.isnan()
            auc = auroc(proba_bin[which_bin], X_bin[which_bin]).item()
        return auc, mse

    def latent_positions(self):
        return None

    def latent_heterogeneity(self):
        return None

    def latent_distance(self, Z):
        return None