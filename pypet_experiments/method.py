from .results import Results
from .data import Dataset
from pypet import ParameterGroup
from typing import Callable, Any
import types
import time
import torch


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
            results = model.results(data.true_values)
            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    log_likelihood=-out[("train", "loss")],
                    X_cts_mse=out[("train", "mse")],
                    X_bin_auroc=out[("train", "auc")],
                    A_auroc=out[("train", "auc_A")],
                    cpu_time=dt,
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop
                ),
                testing_metrics=dict(
                    X_cts_missing_mse=out[("test", "mse")],
                    X_bin_missing_auroc=out[("test", "auc")],
                    A_missing_auroc=out[("test", "auc_A")],
                    edge_density=data.missing_edge_density,
                ),
                estimation_metrics=dict(
                    heteregeneity_l2=results["heteregeneity_l2"],
                    heteregeneity_l2_rel=results["heterogeneity_l2_rel"],
                    latent_ZZt_fro=results["latent_ZZt_fro"],
                    latent_ZZt_fro_rel=results["latent_ZZt_fro_rel"],
                    latent_Proj_fro=results["latent_Proj_fro"],
                    latent_Proj_fro_rel=results["latent_Proj_fro_rel"],
                    bias_l2=results["bias_l2"],
                    bias_l2_rel=results["bias_l2_rel"],
                    weights_BBt_fro=results["weights_BBt_fro"],
                    weights_BBt_fro_rel=results["weights_BBt_fro_rel"],
                    weights_Proj_fro=results["weights_Proj_fro"],
                    weights_Proj_fro_rel=results["weights_Proj_fro_rel"],
                    cts_noise_l2=results["cts_noise_l2"],
                    cts_noise_sqrt_l2=results["cts_noise_sqrt_l2"],
                    cts_noise_log_l2=results["cts_noise_log_l2"],
                    Theta_X_l2=results["Theta_X_l2"],
                    Theta_X_l2_rel=results["Theta_X_l2_rel"],
                    Theta_A_l2=results["Theta_A_l2"],
                    Theta_A_l2_rel=results["Theta_A_l2_rel"],
                    P_l2=results["P_l2"],
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
            results = model.results(data.true_values)
            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    log_likelihood=-out[("train", "loss")],
                    X_cts_mse=out[("train", "mse")],
                    X_bin_auroc=out[("train", "auc")],
                    A_auroc=out[("train", "auc_A")],
                    cpu_time=dt,
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop
                ),
                testing_metrics=dict(
                    X_cts_missing_mse=out[("test", "mse")],
                    X_bin_missing_auroc=out[("test", "auc")],
                    A_missing_auroc=out[("test", "auc_A")],
                    edge_density=data.missing_edge_density,
                ),
                estimation_metrics=dict(
                    heteregeneity_l2=results["heteregeneity_l2"],
                    heteregeneity_l2_rel=results["heterogeneity_l2_rel"],
                    latent_ZZt_fro=results["latent_ZZt_fro"],
                    latent_ZZt_fro_rel=results["latent_ZZt_fro_rel"],
                    latent_Proj_fro=results["latent_Proj_fro"],
                    latent_Proj_fro_rel=results["latent_Proj_fro_rel"],
                    bias_l2=results["bias_l2"],
                    bias_l2_rel=results["bias_l2_rel"],
                    weights_BBt_fro=results["weights_BBt_fro"],
                    weights_BBt_fro_rel=results["weights_BBt_fro_rel"],
                    weights_Proj_fro=results["weights_Proj_fro"],
                    weights_Proj_fro_rel=results["weights_Proj_fro_rel"],
                    cts_noise_l2=results["cts_noise_l2"],
                    cts_noise_sqrt_l2=results["cts_noise_sqrt_l2"],
                    cts_noise_log_l2=results["cts_noise_log_l2"],
                    Theta_X_l2=results["Theta_X_l2"],
                    Theta_X_l2_rel=results["Theta_X_l2_rel"],
                    Theta_A_l2=results["Theta_A_l2"],
                    Theta_A_l2_rel=results["Theta_A_l2_rel"],
                    P_l2=results["P_l2"],
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
            results = model.results(data.true_values)
            dt = time.time() - t0
            return Results(
                training_metrics=dict(
                    log_likelihood=-out[("train", "loss")],
                    X_cts_mse=out[("train", "mse")],
                    X_bin_auroc=out[("train", "auc")],
                    A_auroc=out[("train", "auc_A")],
                    cpu_time=dt,
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop
                ),
                testing_metrics=dict(
                    X_cts_missing_mse=out[("test", "mse")],
                    X_bin_missing_auroc=out[("test", "auc")],
                    A_missing_auroc=out[("test", "auc_A")],
                    edge_density=data.missing_edge_density,
                ),
                estimation_metrics=dict(
                    heteregeneity_l2=results["heteregeneity_l2"],
                    heteregeneity_l2_rel=results["heterogeneity_l2_rel"],
                    latent_ZZt_fro=results["latent_ZZt_fro"],
                    latent_ZZt_fro_rel=results["latent_ZZt_fro_rel"],
                    latent_Proj_fro=results["latent_Proj_fro"],
                    latent_Proj_fro_rel=results["latent_Proj_fro_rel"],
                    bias_l2=results["bias_l2"],
                    bias_l2_rel=results["bias_l2_rel"],
                    weights_BBt_fro=results["weights_BBt_fro"],
                    weights_BBt_fro_rel=results["weights_BBt_fro_rel"],
                    weights_Proj_fro=results["weights_Proj_fro"],
                    weights_Proj_fro_rel=results["weights_Proj_fro_rel"],
                    cts_noise_l2=results["cts_noise_l2"],
                    cts_noise_sqrt_l2=results["cts_noise_sqrt_l2"],
                    cts_noise_log_l2=results["cts_noise_log_l2"],
                    Theta_X_l2=results["Theta_X_l2"],
                    Theta_X_l2_rel=results["Theta_X_l2_rel"],
                    Theta_A_l2=results["Theta_A_l2"],
                    Theta_A_l2_rel=results["Theta_A_l2_rel"],
                    P_l2=results["P_l2"],
                ),
                logs=dict(
                    llk_history=logs[("train", "loss")]
                )
            )
        return cls(model_parameters=model_parameters, fit_function=fit_function)


    @classmethod
    def from_VMP_parameters(cls, model_parameters: ParameterGroup):
        def fit_function(self: Method, data: Dataset, fit_parameters: ParameterGroup):
            from NAIVI import VMP
            from NAIVI.vmp.distributions import Distribution
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            Distribution.set_default_check_args(False)
            fit_parameters_dict = dict(
                max_iter=fit_parameters.vmp.max_iter,
                rel_tol=fit_parameters.vmp.rel_tol,
            )
            t0 = time.time()
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
                edges=data.edges,
                edge_index_left=data.edge_index_left,
                edge_index_right=data.edge_index_right,
            )
            vmp.fit(**fit_parameters_dict)
            dt = time.time() - t0
            results = vmp.evaluate(data.true_values)
            return Results(
                training_metrics=dict(
                    elbo=vmp.elbo_history["sum"][-1],
                    X_cts_mse=results["X_cts_mse"],
                    X_bin_auroc=results["X_bin_auroc"],
                    A_auroc=results["A_auroc"],
                    cpu_time=dt,
                    edge_density=data.edge_density,
                    X_missing_prop=data.covariate_missing_prop,
                    aic=vmp.aic,
                    bic=vmp.bic,
                    gic=vmp.gic,
                ),
                testing_metrics=dict(
                    X_cts_missing_mse=results["X_cts_missing_mse"],
                    X_bin_missing_auroc=results["X_bin_missing_auroc"],
                    A_missing_auroc=results["A_missing_auroc"],
                    edge_density=data.missing_edge_density,
                ),
                estimation_metrics=dict(
                    heteregeneity_l2=results["heteregeneity_l2"],
                    heteregeneity_l2_rel=results["heterogeneity_l2_rel"],
                    latent_ZZt_fro=results["latent_ZZt_fro"],
                    latent_ZZt_fro_rel=results["latent_ZZt_fro_rel"],
                    latent_Proj_fro=results["latent_Proj_fro"],
                    latent_Proj_fro_rel=results["latent_Proj_fro_rel"],
                    bias_l2=results["bias_l2"],
                    bias_l2_rel=results["bias_l2_rel"],
                    weights_BBt_fro=results["weights_BBt_fro"],
                    weights_BBt_fro_rel=results["weights_BBt_fro_rel"],
                    weights_Proj_fro=results["weights_Proj_fro"],
                    weights_Proj_fro_rel=results["weights_Proj_fro_rel"],
                    cts_noise_l2=results["cts_noise_l2"],
                    cts_noise_sqrt_l2=results["cts_noise_sqrt_l2"],
                    cts_noise_log_l2=results["cts_noise_log_l2"],
                    Theta_X_l2=results["Theta_X_l2"],
                    Theta_X_l2_rel=results["Theta_X_l2_rel"],
                    Theta_A_l2=results["Theta_A_l2"],
                    Theta_A_l2_rel=results["Theta_A_l2_rel"],
                    P_l2=results["P_l2"],
                ),
                logs=dict(
                    elbo_history=vmp.elbo_history["sum"]
                )
            )

        return cls(model_parameters=model_parameters, fit_function=fit_function)
