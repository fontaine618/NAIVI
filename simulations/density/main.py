import os
import sys

SIM_NAME = "density"
SIM_PATH = "/home/simon/Documents/NAIVI/simulations/" + SIM_NAME

from pypet import Environment
from pypet.utils.explore import cartesian_product
import numpy as np
from simulations.gen_data import generate_dataset
from NAIVI.vmp_tf.vmp.joint_model2 import JointModel2
import pandas as pd
import tensorflow as tf


def run(traj):
    try:
        # extract parameters
        N = traj.par.data.N
        K = traj.par.data.K
        p_cts = traj.par.data.p_cts
        p_bin = traj.par.data.p_bin
        var_adj = traj.par.data.var_adj
        var_cov = traj.par.data.var_cov
        missing_rate = traj.par.data.missing_rate
        seed = traj.par.data.seed
        alpha_mean = traj.par.data.alpha_mean

        print(p_bin, alpha_mean, seed)

        adj_model = traj.par.model.adj_model
        bin_model = traj.par.model.bin_model

        # generate data
        Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, A, B, B0 = generate_dataset(
            N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_adj=var_adj, alpha_mean=alpha_mean,
            var_cov=var_cov, missing_rate=missing_rate, seed=seed,
            link_model=adj_model, bin_model=bin_model
        )

        density = tf.reduce_sum(tf.where(tf.math.is_nan(A), 0., A)).numpy() * 2 / (N*(N-1))
        density = density
    except:
        density = np.nan
    traj.f_add_result("runs.$.alpha_mean", alpha_mean, "alpha_mean")
    traj.f_add_result("runs.$.seed", seed, "seed")
    traj.f_add_result("runs.$.density", density, "Network density")

    print(p_bin, alpha_mean, seed, density)
    return p_bin, alpha_mean, seed, density


def post_processing(traj, result_list):

    run_idx = [res[0] for res in result_list]
    p_bin = [res[1][0] for res in result_list]
    alpha_mean = [res[1][1] for res in result_list]
    seed = [res[1][2] for res in result_list]
    density = [res[1][3] for res in result_list]

    df = pd.DataFrame({
        "alpha_mean": [alpha_mean[i] for i in run_idx],
        "seed": [seed[i] for i in run_idx],
        "p_bin": [p_bin[i] for i in run_idx],
        "density": [density[i] for i in run_idx]
    }, index=run_idx)
    print(df)
    traj.f_add_result("data_frame", df, "Summary across replications")
    df.to_csv(SIM_PATH + "/results/summary.csv")


def main():
    # pypet environment
    env = Environment(
        trajectory=SIM_NAME,
        comment="Get density from alpha mean",
        log_config=None,
        multiproc=False,
        ncores=1,
        filename=SIM_PATH + "/results/",
        overwrite_file=True
    )
    traj = env.trajectory

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
        "data.p_bin", np.int64(100), "Number of binary covariates"
    )
    traj.f_add_parameter(
        "data.var_adj", np.float64(1.), "True variance in the link Probit model"
    )
    traj.f_add_parameter(
        "data.var_cov", np.float64(1.), "True variance in the covariate model (cts and bin)"
    )
    traj.f_add_parameter(
        "data.missing_rate", np.float64(0.1), "Missing rate"
    )
    traj.f_add_parameter(
        "data.seed", np.int64(1), "Random seed"
    )
    traj.f_add_parameter(
        "data.alpha_mean", np.float64(-1.85), "Mean of the heterogeneity parameter"
    )

    # parameters (model)
    traj.f_add_parameter(
        "model.K", np.int64(5), "Number of latent components in the model"
    )
    traj.f_add_parameter(
        "model.adj_model", "Logistic", "Adjacency model"
    )
    traj.f_add_parameter(
        "model.bin_model", "Logistic", "Binary covariate model"
    )

    # parameters (fit)
    traj.f_add_parameter(
        "fit.n_iter", np.int64(20), "Number of VEM iterations"
    )
    traj.f_add_parameter(
        "fit.n_vmp", np.int64(5), "Number of VMP iterations per E-step"
    )
    traj.f_add_parameter(
        "fit.n_gd", np.int64(5), "Number of GD iterations per M-step"
    )
    traj.f_add_parameter(
        "fit.step_size", np.float64(0.01), "GD Step size"
    )

    # experiment
    explore_dict = {
        "data.alpha_mean": np.array(
            [-3.2, -2.8, -2.4, -2., -1.6, -1.2, -0.8, -0.4, 0.0, 0.4]
        ),
        "data.seed": np.arange(0, 100, 1)
    }
    experiment = cartesian_product(explore_dict, tuple(explore_dict.keys()))
    traj.f_explore(experiment)

    env.add_postprocessing(post_processing)
    env.run(run)
    env.disable_logging()


if __name__ == "__main__":
    main()