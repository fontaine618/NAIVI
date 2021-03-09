import os
import sys

PATH = "/home/simfont/scratch/NAIVI/"
sys.path.append(PATH)
sys.path.append(PATH + "NAIVI/")
sys.path.append(PATH + "venv/")
sys.path.append(PATH + "venv/lib/")
SIM_NAME = "networksize_binary"
SIM_PATH = PATH + "/simulations/" + SIM_NAME

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

        print(p_bin, N, seed)

        K_model = traj.par.model.K
        adj_model = traj.par.model.adj_model
        bin_model = traj.par.model.bin_model

        n_iter = traj.par.fit.n_iter
        n_vmp = traj.par.fit.n_vmp
        n_gd = traj.par.fit.n_gd
        step_size = traj.par.fit.step_size

        # generate data
        Z, alpha, X_cts, X_cts_missing, X_bin, X_bin_missing, A, B, B0 = generate_dataset(
            N=N, K=K, p_cts=p_cts, p_bin=p_bin, var_adj=var_adj, alpha_mean=alpha_mean,
            var_cov=var_cov, missing_rate=missing_rate, seed=seed,
            link_model=adj_model, bin_model=bin_model
        )
        initial = {
            "bias": B0,
            "weights": B,
            "positions": Z,
            "heterogeneity": alpha
        }

        # initialize model
        model = JointModel2(
            K=K_model,
            A=A,
            X_cts=X_cts,
            X_bin=X_bin,
            link_model="Logistic",
            bin_model="Logistic",
            initial=initial
        )

        # fit
        model.fit(
            n_iter=n_iter,
            n_vmp=n_vmp,
            n_gd=n_gd,
            verbose=False,
        )

        # metrics
        metrics = model.covariate_metrics(X_cts_missing, X_bin_missing)
        dists = model.latent_distance(Z)
        elbo = model.elbo.numpy()
        density = tf.reduce_sum(tf.where(tf.math.is_nan(A), 0., A)).numpy() * 2 / (N*(N-1))

        # store result
        elbo = elbo
        mse = metrics["mse"]
        auroc = metrics["auroc"]
        dist_inv = dists["inv"]
        dist_proj = dists["proj"]
        density = density
    except:
        elbo = np.nan
        mse = np.nan
        auroc = np.nan
        dist_inv = np.nan
        dist_proj = np.nan
        density = np.nan
    traj.f_add_result("runs.$.p_bin", p_bin, "p_bin")
    traj.f_add_result("runs.$.N", N, "N")
    traj.f_add_result("runs.$.seed", seed, "seed")
    traj.f_add_result("runs.$.elbo", elbo, "ELBO")
    traj.f_add_result("runs.$.mse", mse, "MSE")
    traj.f_add_result("runs.$.auroc", auroc, "AUROC")
    traj.f_add_result("runs.$.dist_inv", dist_inv, "Invariant distance")
    traj.f_add_result("runs.$.dist_proj", dist_proj, "Projection distance")
    traj.f_add_result("runs.$.density", density, "Network density")

    print(p_bin, N, seed, elbo, mse, auroc, dist_inv, dist_proj, density)
    return p_bin, N, seed, elbo, mse, auroc, dist_inv, dist_proj, density


def post_processing(traj, result_list):

    run_idx = [res[0] for res in result_list]
    p_bin = [res[1][0] for res in result_list]
    N = [res[1][1] for res in result_list]
    seed = [res[1][2] for res in result_list]
    elbo = [res[1][3] for res in result_list]
    mse = [res[1][4] for res in result_list]
    auroc = [res[1][5] for res in result_list]
    dist_inv = [res[1][6] for res in result_list]
    dist_proj = [res[1][7] for res in result_list]
    density = [res[1][8] for res in result_list]

    df = pd.DataFrame({
        "N": [N[i] for i in run_idx],
        "seed": [seed[i] for i in run_idx],
        "p_bin": [p_bin[i] for i in run_idx],
        "elbo": [elbo[i] for i in run_idx],
        "mse": [mse[i] for i in run_idx],
        "auroc": [auroc[i] for i in run_idx],
        "dist_inv": [dist_inv[i] for i in run_idx],
        "dist_proj": [dist_proj[i] for i in run_idx],
        "density": [density[i] for i in run_idx]
    }, index=run_idx)
    print(df)
    traj.f_add_result("data_frame", df, "Summary across replications")
    df.to_csv(SIM_PATH + "/results/summary.csv")


def main():
    # pypet environment
    env = Environment(
        trajectory=SIM_NAME,
        comment="Experiment on network size with binary covariates",
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
        "data.p_bin", np.int64(0), "Number of binary covariates"
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
        "data.N": np.array(
            [50, 100, 200, 500, 1000, 2000]
        ),
        "data.p_bin": np.array([10, 100, 500]),
        "data.seed": np.arange(0, 100, 1)
    }
    experiment = cartesian_product(explore_dict, tuple(explore_dict.keys()))
    traj.f_explore(experiment)

    env.add_postprocessing(post_processing)
    env.run(run)
    env.disable_logging()


if __name__ == "__main__":
    main()