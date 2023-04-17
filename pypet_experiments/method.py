from .results import Results
from .data import Dataset
from .method_output import MethodOutput
from .metrics import Metrics
from pypet import ParameterGroup
from typing import Callable, Any
import types
import time
import torch
from torchmetrics.functional import auroc, mean_squared_error
import gc


class Method:
    """
    This is a bit convoluted, but here is how it works.

    We really only directly call from_parameters, which then calls the appropriate
    from_{algo}_parameters method. This is done to allow for different structures.

    Then, we have a fit function that is bound to the method instance. This is done
    to allow for different routines.

    For each new routine, we need to add a new from_{algo}_parameters method,
    which needs to get all appropriate parameters from the trajectory
    and define the fit_function. This fit function will have acess to self, and
    in particular to the parameters extracted in the from_{algo}_parameters method.
    """

    def __init__(
            self,
            model_parameters: ParameterGroup,
            fit_function: Callable[["Method", Dataset, ParameterGroup], Results]
    ):
        self.model_parameters = model_parameters
        self.model: Any = None
        self.fit = types.MethodType(fit_function, self)

    @classmethod
    def from_parameters(cls, method: str, model_parameters: ParameterGroup):
        method_name = f"from_{method}_parameters"
        if hasattr(cls, method_name):
            return getattr(cls, method_name)(model_parameters)
        else:
            raise NotImplementedError(f"Method {method} not implemented")


    @classmethod
    def from_MAP_parameters(cls, model_parameters: ParameterGroup):
        def fit_function(self: Method, data: Dataset, fit_parameters: ParameterGroup):
            from NAIVI import MAP
            from NAIVI.initialization import initialize
            from NAIVI.utils.data import JointDataset
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

            train = JointDataset(
                i0=data.edge_index_left,
                i1=data.edge_index_right,
                A=data.edges,
                X_cts=data.continuous_covariates,
                X_bin=data.binary_covariates,
                return_missingness=False,
                cuda=True
            )
            test = JointDataset(
                i0=data.edge_index_left,
                i1=data.edge_index_right,
                A=data.edges_missing,
                X_cts=data.continuous_covariates_missing,
                X_bin=data.binary_covariates_missing,
                return_missingness=False,
                test=True,
                cuda=True
            )
            fit_parameters_dict = dict(
                max_iter=fit_parameters.map.max_iter,
                eps=fit_parameters.map.eps,
                optimizer=fit_parameters.map.optimizer,
                lr=fit_parameters.map.lr,
                train=train,
                test=test,
                return_log=True
            )
            model_args = {
                "K": self.model_parameters["latent_dim"],
                "N": data.n_nodes,
                "p_cts": data.p_cts,
                "p_bin": data.p_bin,
                "mnar": False,
                "network_weight": 1.0,
                "position_prior": (
                    self.model_parameters["latent_prior_mean"],
                    self.model_parameters["latent_prior_variance"]
                ),
                "heterogeneity_prior": (
                    self.model_parameters["heterogeneity_prior_mean"],
                    self.model_parameters["heterogeneity_prior_variance"]
                ),
                "estimate_components": False
            }
            t0 = time.time()
            model = MAP(**model_args)
            init = initialize(train, self.model_parameters["latent_dim"])
            model.init(**init)
            out, logs = model.fit(**fit_parameters_dict)
            model_output = MethodOutput(**model.output(train))
            metrics = Metrics(model_output, data).metrics
            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    log_likelihood=-out[("train", "loss")],
                    cpu_time=dt,
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop,
                    **metrics["training"]
                ),
                testing_metrics=dict(
                    edge_density=data.missing_edge_density,
                    **metrics["testing"]
                ),
                estimation_metrics=dict(
                    **metrics["estimation"]
                ),
                logs=dict(
                    llk_history=logs[("train", "loss")]
                )
            )
        return cls(model_parameters=model_parameters, fit_function=fit_function)


    @classmethod
    def from_MLE_parameters(cls, model_parameters: ParameterGroup):
        def fit_function(self: Method, data: Dataset, fit_parameters: ParameterGroup):
            from NAIVI import MLE
            from NAIVI.initialization import initialize
            from NAIVI.utils.data import JointDataset
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

            train = JointDataset(
                i0=data.edge_index_left,
                i1=data.edge_index_right,
                A=data.edges,
                X_cts=data.continuous_covariates,
                X_bin=data.binary_covariates,
                return_missingness=False,
                cuda=True
            )
            test = JointDataset(
                i0=data.edge_index_left,
                i1=data.edge_index_right,
                A=data.edges_missing,
                X_cts=data.continuous_covariates_missing,
                X_bin=data.binary_covariates_missing,
                return_missingness=False,
                test=True,
                cuda=True
            )
            fit_parameters_dict = dict(
                max_iter=fit_parameters.map.max_iter,
                eps=fit_parameters.map.eps,
                optimizer=fit_parameters.map.optimizer,
                lr=fit_parameters.map.lr,
                train=train,
                test=test,
                return_log=True
            )
            model_args = {
                "K": self.model_parameters["latent_dim"],
                "N": data.n_nodes,
                "p_cts": data.p_cts,
                "p_bin": data.p_bin,
                "mnar": False,
                "network_weight": 1.0,
                "position_prior": (
                    self.model_parameters["latent_prior_mean"],
                    self.model_parameters["latent_prior_variance"]
                ),
                "heterogeneity_prior": (
                    self.model_parameters["heterogeneity_prior_mean"],
                    self.model_parameters["heterogeneity_prior_variance"]
                ),
                "estimate_components": False
            }
            t0 = time.time()
            model = MLE(**model_args)
            init = initialize(train, self.model_parameters["latent_dim"])
            model.init(**init)
            out, logs = model.fit(**fit_parameters_dict)
            model_output = MethodOutput(**model.output(train))
            metrics = Metrics(model_output, data).metrics
            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    log_likelihood=-out[("train", "loss")],
                    cpu_time=dt,
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop,
                    **metrics["training"]
                ),
                testing_metrics=dict(
                    edge_density=data.missing_edge_density,
                    **metrics["testing"]
                ),
                estimation_metrics=dict(
                    **metrics["estimation"]
                ),
                logs=dict(
                    llk_history=logs[("train", "loss")]
                )
            )
        return cls(model_parameters=model_parameters, fit_function=fit_function)


    @classmethod
    def from_ADVI_parameters(cls, model_parameters: ParameterGroup):
        def fit_function(self: Method, data: Dataset, fit_parameters: ParameterGroup):
            from NAIVI import ADVI
            from NAIVI.initialization import initialize
            from NAIVI.utils.data import JointDataset
            torch.set_default_tensor_type(torch.cuda.FloatTensor)

            train = JointDataset(
                i0=data.edge_index_left,
                i1=data.edge_index_right,
                A=data.edges,
                X_cts=data.continuous_covariates,
                X_bin=data.binary_covariates,
                return_missingness=False,
                cuda=True
            )
            test = JointDataset(
                i0=data.edge_index_left,
                i1=data.edge_index_right,
                A=data.edges_missing,
                X_cts=data.continuous_covariates_missing,
                X_bin=data.binary_covariates_missing,
                return_missingness=False,
                test=True,
                cuda=True
            )
            fit_parameters_dict = dict(
                max_iter=fit_parameters.advi.max_iter,
                eps=fit_parameters.advi.eps,
                optimizer=fit_parameters.advi.optimizer,
                lr=fit_parameters.advi.lr,
                train=train,
                test=test,
                return_log=True
            )
            model_args = {
                "K": self.model_parameters["latent_dim"],
                "N": data.n_nodes,
                "p_cts": data.p_cts,
                "p_bin": data.p_bin,
                "mnar": False,
                "network_weight": 1.0,
                "position_prior": (
                    self.model_parameters["latent_prior_mean"],
                    self.model_parameters["latent_prior_variance"]
                ),
                "heterogeneity_prior": (
                    self.model_parameters["heterogeneity_prior_mean"],
                    self.model_parameters["heterogeneity_prior_variance"]
                ),
                "estimate_components": False
            }
            t0 = time.time()
            model = ADVI(**model_args)
            init = initialize(train, self.model_parameters["latent_dim"])
            model.init(**init)
            out, logs = model.fit(**fit_parameters_dict)
            model_output = MethodOutput(**model.output(train))
            metrics = Metrics(model_output, data).metrics
            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    log_likelihood=-out[("train", "loss")],
                    cpu_time=dt,
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop,
                    **metrics["training"]
                ),
                testing_metrics=dict(
                    edge_density=data.missing_edge_density,
                    **metrics["testing"]
                ),
                estimation_metrics=dict(
                    **metrics["estimation"]
                ),
                logs=dict(
                    llk_history=logs[("train", "loss")]
                )
            )
        return cls(model_parameters=model_parameters, fit_function=fit_function)


    @classmethod
    def from_VMP_parameters(cls, model_parameters: ParameterGroup, covariates_only: bool = False):
        def fit_function(self: Method, data: Dataset, fit_parameters: ParameterGroup):
            from NAIVI import VMP
            from NAIVI.vmp import enable_logging
            from NAIVI.vmp.distributions import Distribution
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            Distribution.set_default_check_args(False)
            # Distribution.set_default_check_args(True)
            # enable_logging()
            fit_parameters_dict = dict(
                max_iter=fit_parameters.vmp.max_iter,
                rel_tol=fit_parameters.vmp.rel_tol,
            )
            cv_folds = fit_parameters.vmp.cv_folds
            t0 = time.time()
            if cv_folds > 1:
                from NAIVI.vmp import CVVMP
                cv_vmp = CVVMP(
                    latent_dim=self.model_parameters.latent_dim,
                    n_nodes=data.n_nodes,
                    heterogeneity_prior_mean=self.model_parameters.heterogeneity_prior_mean,
                    heterogeneity_prior_variance=self.model_parameters.heterogeneity_prior_variance,
                    latent_prior_mean=self.model_parameters.latent_prior_mean,
                    latent_prior_variance=self.model_parameters.latent_prior_variance,
                    # data
                    binary_covariates=data.binary_covariates,
                    continuous_covariates=data.continuous_covariates,
                    edges=data.edges if not covariates_only else None,
                    edge_index_left=data.edge_index_left,
                    edge_index_right=data.edge_index_right,
                    folds=cv_folds,
                    logistic_approximation=self.model_parameters.vmp.logistic_approximation,
                )
                cv_vmp.fit(**fit_parameters_dict)
                covariate_elbo = cv_vmp.covariate_elbo
                covariate_log_likelihood = cv_vmp.covariate_log_likelihood
            vmp = VMP(
                # dimension and model parameters
                latent_dim=self.model_parameters.latent_dim,
                n_nodes=data.n_nodes,
                heterogeneity_prior_mean=self.model_parameters.heterogeneity_prior_mean,
                heterogeneity_prior_variance=self.model_parameters.heterogeneity_prior_variance,
                latent_prior_mean=self.model_parameters.latent_prior_mean,
                latent_prior_variance=self.model_parameters.latent_prior_variance,
                # data
                binary_covariates=data.binary_covariates,
                continuous_covariates=data.continuous_covariates,
                edges=data.edges if not covariates_only else None,
                edge_index_left=data.edge_index_left,
                edge_index_right=data.edge_index_right,
                logistic_approximation=self.model_parameters.vmp.logistic_approximation,
            )
            vmp.fit(**fit_parameters_dict)
            model_output = MethodOutput(**vmp.output())
            metrics = Metrics(model_output, data).metrics
            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    elbo=vmp.elbo_history["sum"][-1],
                    cpu_time=dt,
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop,
                    elbo_covariates=vmp.elbo_covariates,
                    aic=vmp.aic,
                    bic=vmp.bic,
                    weights_entropy=vmp.weights_entropy,
                    cv_covariate_elbo=covariate_elbo if cv_folds > 1 else 0.,
                    cv_covariate_log_likelihood=covariate_log_likelihood if cv_folds > 1 else 0.,
                    **metrics["training"]
                ),
                testing_metrics=dict(
                    edge_density=data.missing_edge_density,
                    **metrics["testing"]
                ),
                estimation_metrics=dict(
                    **metrics["estimation"]
                ),
                logs=dict(
                    elbo_history=vmp.elbo_history["sum"]
                )
            )
        return cls(model_parameters=model_parameters, fit_function=fit_function)

    @classmethod
    def from_Oracle_parameters(cls, model_parameters: ParameterGroup):
        def fit_function(self: Method, data: Dataset, fit_parameters: ParameterGroup):
            if "Theta_X" not in data.true_values:
                raise ValueError("Oracle method requires true values for Theta_X")
            t0 = time.time()
            theta_X = data.true_values["Theta_X"]
            theta_A = data.true_values["Theta_A"]
            proba_edges = data.true_values["P"]
            p_cts = data.continuous_covariates.shape[1]
            p_bin = data.binary_covariates.shape[1]
            mean_cts, logit_bin = theta_X[:, :p_cts], theta_X[:, p_cts:]
            model_output = MethodOutput(
                pred_continuous_covariates=mean_cts,
                pred_binary_covariates=torch.sigmoid(logit_bin),
                pred_edges=proba_edges,
                linear_predictor_covariates=theta_X,
                linear_predictor_edges=theta_A
            )
            metrics = Metrics(model_output, data).metrics
            dt = time.time() - t0

            return Results(
                training_metrics=dict(
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop,
                    **metrics["training"]
                ),
                testing_metrics=dict(
                    edge_density=data.missing_edge_density,
                    **metrics["testing"]
                ),
                estimation_metrics=dict(),
                logs=dict()
            )
        return cls(model_parameters=model_parameters, fit_function=fit_function)


    @classmethod
    def from_Mean_parameters(cls, model_parameters: ParameterGroup):
        def fit_function(self: Method, data: Dataset, fit_parameters: ParameterGroup):
            from NAIVI import Mean
            t0 = time.time()
            mean = Mean(
                binary_covariates=data.binary_covariates,
                continuous_covariates=data.continuous_covariates,
            )
            cts_mean = mean.cts_mean.repeat(data.n_nodes, 1)
            bin_mean = mean.bin_mean.repeat(data.n_nodes, 1)
            theta_X = torch.cat([cts_mean, torch.logit(bin_mean)], dim=1)
            model_output = MethodOutput(
                pred_continuous_covariates=cts_mean,
                pred_binary_covariates=bin_mean,
                linear_predictor_covariates=theta_X
            )
            metrics = Metrics(model_output, data).metrics
            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop,
                    cpu_time=dt,
                    **metrics["training"]
                ),
                testing_metrics=dict(
                    edge_density=data.missing_edge_density,
                    **metrics["testing"]
                ),
                estimation_metrics=dict(),
                logs=dict()
            )

        return cls(model_parameters=model_parameters, fit_function=fit_function)

    @classmethod
    def from_NetworkSmoothing_parameters(cls, model_parameters: ParameterGroup):
        def fit_function(self: Method, data: Dataset, fit_parameters: ParameterGroup):
            from NAIVI import NetworkSmoothing
            t0 = time.time()
            binary_proba, continuous_mean = NetworkSmoothing().fit(
                binary_covariates=data.binary_covariates,
                continuous_covariates=data.continuous_covariates,
                edges=data.edges,
                edge_index_left=data.edge_index_left,
                edge_index_right=data.edge_index_right,
            )
            theta_X = torch.cat([continuous_mean, binary_proba], dim=1)
            model_output = MethodOutput(
                pred_continuous_covariates=continuous_mean,
                pred_binary_covariates=binary_proba,
                linear_predictor_covariates=theta_X
            )
            metrics = Metrics(model_output, data).metrics
            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop,
                    cpu_time=dt,
                    **metrics["training"]
                ),
                testing_metrics=dict(
                    edge_density=data.missing_edge_density,
                    **metrics["testing"]
                ),
                estimation_metrics=dict(),
                logs=dict()
            )

        return cls(model_parameters=model_parameters, fit_function=fit_function)

    @classmethod
    def from_MICE_parameters(cls, model_parameters: ParameterGroup):
        def fit_function(self: Method, data: Dataset, fit_parameters: ParameterGroup):
            from NAIVI import MICE
            t0 = time.time()
            mice = MICE(
                binary_covariates=data.binary_covariates,
                continuous_covariates=data.continuous_covariates,
                max_iter=fit_parameters.mice.max_iter,
                rel_tol=fit_parameters.mice.rel_tol
            )
            if mice.p_cts + mice.p_bin > 250: # this is too long, so we skip it
                return Results(
                    training_metrics=dict(
                        edge_density=data.edge_density,
                        X_missing_prop=data.covariate_missing_prop
                    ),
                    testing_metrics=dict(
                        edge_density=data.missing_edge_density
                    ),
                    estimation_metrics=dict(),
                    logs=dict()
                )
            binary_proba, continuous_mean = mice.fit_transform()
            theta_X = torch.cat([continuous_mean, binary_proba], dim=1)
            model_output = MethodOutput(
                pred_continuous_covariates=continuous_mean,
                pred_binary_covariates=binary_proba,
                linear_predictor_covariates=theta_X
            )
            metrics = Metrics(model_output, data).metrics

            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop,
                    cpu_time=dt,
                    **metrics["training"]
                ),
                testing_metrics=dict(
                    edge_density=data.missing_edge_density,
                    **metrics["testing"]
                ),
                estimation_metrics=dict(),
                logs=dict()
            )

        return cls(model_parameters=model_parameters, fit_function=fit_function)

    @classmethod
    def from_KNN_parameters(cls, model_parameters: ParameterGroup):
        def fit_function(self: Method, data: Dataset, fit_parameters: ParameterGroup):
            from NAIVI import KNN
            t0 = time.time()
            knn = KNN(
                binary_covariates=data.binary_covariates,
                continuous_covariates=data.continuous_covariates,
                n_neighbors=fit_parameters.knn.n_neighbors
            )
            binary_proba, continuous_mean = knn.fit_transform()
            theta_X = torch.cat([continuous_mean, binary_proba], dim=1)
            model_output = MethodOutput(
                pred_continuous_covariates=continuous_mean,
                pred_binary_covariates=binary_proba,
                linear_predictor_covariates=theta_X
            )
            metrics = Metrics(model_output, data).metrics
            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop,
                    cpu_time=dt,
                    **metrics["training"]
                ),
                testing_metrics=dict(
                    edge_density=data.missing_edge_density,
                    **metrics["testing"]
                ),
                estimation_metrics=dict(),
                logs=dict()
            )

        return cls(model_parameters=model_parameters, fit_function=fit_function)

    @classmethod
    def from_FA_parameters(cls, model_parameters: ParameterGroup):
        return cls.from_VMP_parameters(model_parameters, True)

