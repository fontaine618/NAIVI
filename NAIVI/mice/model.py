import torch
import numpy as np
from torchmetrics.functional import auroc, mean_squared_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


class MICE:

    def __init__(self, K, N, p_cts, p_bin):
        estimator = BayesianRidge()
        self.model = IterativeImputer(
            random_state=0, estimator=estimator, imputation_order="random", max_iter=100,
            verbose=2, skip_complete=True, tol=0.001
        )
        self.p_cts = p_cts
        self.p_bin = p_bin
        self.N = N

    def fit(self, train, test=None, **kwargs):
        # get train data
        _, _, _, _, X_cts, X_bin = train[:]
        if X_cts is None:
            X_cts = torch.zeros((self.N, 0))
        if X_bin is None:
            X_bin = torch.zeros((self.N, 0))
        # concatenate
        X = torch.cat([X_cts, X_bin], 1)

        # predict
        X_pred = self.model.fit_transform(X.cpu())
        X_cts_pred, X_bin_pred, _ = np.split(
            X_pred, indices_or_sections=[self.p_cts, self.p_cts + self.p_bin], axis=1
        )

        # get test data
        _, _, _, _, X_test_cts, X_test_bin = test[:]
        if self.p_cts > 0:
            X_cts_pred[np.isnan(X_test_cts) == 1.0] = np.nan
            X_cts_pred = torch.tensor(X_cts_pred)
        if self.p_bin > 0:
            X_bin_pred[np.isnan(X_test_bin) == 1.0] = np.nan
            X_bin_pred = X_bin_pred.clip(0.0, 1.0)
            X_bin_pred = torch.tensor(X_bin_pred)

        out = self.metrics(X_test_bin, X_test_cts, X_cts_pred, X_bin_pred)
        return out

    def metrics(self, X_bin, X_cts, mean_cts, proba_bin, epoch=None):
        auroc_test, mse_test = self.prediction_metrics(
            X_bin, X_cts, mean_cts, proba_bin
        )
        out = {("test", "mse"): mse_test, ("test", "auc"): auroc_test}
        return out

    def prediction_metrics(self, X_bin, X_cts, mean_cts, proba_bin):
        mse = 0.0
        auc = 0.0
        if X_cts is not None:
            which_cts = ~X_cts.isnan()
            mse = mean_squared_error(mean_cts[which_cts], X_cts[which_cts]).item()
        if X_bin is not None:
            which_bin = ~X_bin.isnan()
            auc = auroc(proba_bin[which_bin], X_bin[which_bin].int()).item()
        return auc, mse
