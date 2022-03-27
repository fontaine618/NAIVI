import numpy as np
import os
from pypet import Environment
from pypet.utils.explore import cartesian_product
from NAIVI_experiments.post_processing import post_processing
from NAIVI_experiments.run import run


def main(path, name, which, explore_dict):
    path = path + name + "/"
    os.makedirs(path, exist_ok=True)
    comment = "\n".join([
        "{:<20} {}".format(k, v) for k, v in explore_dict.items()
    ])
    print(comment)
    # pypet environment
    env = Environment(
        trajectory=name,
        comment=comment,
        log_config=None,
        multiproc=False,
        ncores=0,
        filename=path + which,
        overwrite_file=True
    )
    traj = env.trajectory
    traj.f_add_parameter(
        "path", path, "Path"
    )

    # parameters (data generation)
    traj.f_add_parameter(
        "data.N", np.int64(500), "Number of nodes"
    )
    traj.f_add_parameter(
        "data.K", np.int64(5), "True number of latent components"
    )
    traj.f_add_parameter(
        "data.p_cts", np.int64(0), "Number of continuous covariates"
    )
    traj.f_add_parameter(
        "data.p_bin", np.int64(0), "Number of binary covariates"
    )
    traj.f_add_parameter(
        "data.var_cov", np.float64(1.), "True variance in the covariate model (cts and bin)"
    )
    traj.f_add_parameter(
        "data.missing_mean", np.float64(-10000.0), "Missing mean"
    )
    traj.f_add_parameter(
        "data.seed", np.int64(0), "Random seed"
    )
    traj.f_add_parameter(
        "data.alpha_mean", np.float64(-1.85), "Mean of the heterogeneity parameter for generating data"
    )
    traj.f_add_parameter(
        "data.mnar_sparsity", np.float64(1.0), "Proportion of attributes with MCAR"
    )
    traj.f_add_parameter(
        "data.adjacency_noise", np.float64(0.0), "variance of gaussian error added to logit probability"
    )
    traj.f_add_parameter(
        "data.constant_components", np.bool(True), "W=1 in ZWZ', otherwise exp(N(0,4)) sorted"
    )

    # parameters (model)
    traj.f_add_parameter(
        "model.K", np.int64(5), "Number of latent components in the model"
    )
    traj.f_add_parameter(
        "model.mnar", np.bool(False), "Whether to fit MNAR or not"
    )
    traj.f_add_parameter(
        "model.alpha_mean", np.float64(-1.85), "Mean of the heterogeneity parameter"
    )
    traj.f_add_parameter(
        "model.reg", np.float64(0.), "regularization parameter for mnar"
    )
    traj.f_add_parameter(
        "model.network_weight", np.float64(1.0), "weight of the network in the objective"
    )
    traj.f_add_parameter(
        "model.estimate_components", np.bool(False), "whether to estimate W in ZWZ'"
    )

    # parameters (fit)
    traj.f_add_parameter(
        "fit.keep_logs", np.bool(True), "Whether to store logs"
    )
    traj.f_add_parameter(
        "fit.algo", "MLE", "Inference algorithm"
    )
    traj.f_add_parameter(
        "fit.max_iter", np.int64(200), "Number of iterations"
    )
    traj.f_add_parameter(
        "fit.n_sample", np.int64(0), "Number of samples for VIMC; 0 defaults to 200/N"
    )
    traj.f_add_parameter(
        "fit.mcmc_n_sample", np.int64(5000), "Number of samples for MCMC per chain"
    )
    traj.f_add_parameter(
        "fit.mcmc_n_chains", np.int64(10), "Number of samples for MCMC"
    )
    traj.f_add_parameter(
        "fit.mcmc_n_warmup", np.int64(1000), "Number of warmup samples for MCMC per chain"
    )
    traj.f_add_parameter(
        "fit.mcmc_n_thin", np.int64(10), "Thinning frequency"
    )
    traj.f_add_parameter(
        "fit.eps", np.float64(0.0005), "convergence threshold"
    )
    traj.f_add_parameter(
        "fit.lr", np.float64(0.01), "GD Step size"
    )
    traj.f_add_parameter(
        "fit.power", np.float64(0.0), "GD Step size decrement (t^-power)"
    )
    traj.f_add_parameter(
        "fit.init", "random", "initialization method"
    )
    traj.f_add_parameter(
        "fit.optimizer", "Adam", "optimizer choice"
    )

    # experiment
    experiment = cartesian_product(explore_dict, tuple(explore_dict.keys()))
    traj.f_explore(experiment)
    env.add_postprocessing(post_processing)
    env.run(run)
    env.disable_logging()