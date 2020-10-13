from pypet import Environment
from pypet.utils.explore import cartesian_product
from pypet import pypetconstants
import numpy as np
from simulations.gen_data import generate_dataset
from models.vmp.joint_model2 import JointModel2
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

        print(p_cts, missing_rate, seed)

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
            K=K,
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
            # X_cts_missing=X_cts_missing,
            # X_bin_missing=X_bin_missing,
            # positions_true=Z
        )

        # metrics
        metrics = model.covariate_metrics(X_cts_missing, X_bin_missing)
        dists = model.latent_distance(Z)
        elbo = model.elbo.numpy()
        density = tf.reduce_sum(tf.where(tf.math.is_nan(A), 0., A)).numpy() * 2 / (N*(N-1))

        # store result
        traj.f_add_result("runs.$.elbo", elbo, "ELBO")
        traj.f_add_result("runs.$.mse", metrics["mse"], "MSE")
        traj.f_add_result("runs.$.auroc", metrics["auroc"], "AUROC")
        traj.f_add_result("runs.$.dist_inv", dists["inv"], "Invariant distance")
        traj.f_add_result("runs.$.dist_proj", dists["proj"], "Projection distance")
        traj.f_add_result("runs.$.density", density, "Network density")
    except:
        traj.f_add_result("runs.$.elbo", np.nan, "ELBO")
        traj.f_add_result("runs.$.mse", np.nan, "MSE")
        traj.f_add_result("runs.$.auroc", np.nan, "AUROC")
        traj.f_add_result("runs.$.dist_inv", np.nan, "Invariant distance")
        traj.f_add_result("runs.$.dist_proj", np.nan, "Projection distance")
        traj.f_add_result("runs.$.density", np.nan, "Network density")
        traj.f_add_result("runs.$.error", 1., "Network density")


def post_processing(traj, result_list):
    seed_range = traj.par.data.f_get("seed").f_get_range()
    missing_rate_range = traj.par.data.f_get("missing_rate").f_get_range()
    elbo = [res[1] for res in result_list]
    run_idx = [res[0] for res in result_list]
    df = pd.DataFrame({
        "missing_rate": [missing_rate_range[i] for i in run_idx],
        "seed": [seed_range[i] for i in run_idx],
        "elbo": [elbo[i] for i in run_idx]
    })
    print(df)
    summary = df.groupby("missing_rate").mean()
    print(summary)
    traj.f_add_result("summary", summary, "Summary across replications")


def main():
    # pypet environment
    env = Environment(
        trajectory="missing_rate",
        comment="Test experiment with varying missing rate",
        log_config=None,
        multiproc=True,
        ncores=10,
        use_pool=True,
        freeze_input=True,
        wrap_mode=pypetconstants.WRAP_MODE_QUEUE,
        graceful_exit=True,
        filename="./simulations/results/test/",
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
        "data.p_cts", np.int64(10), "Number of continuous covariates"
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
        "data.missing_rate", np.float64(0.2), "Missing rate"
    )
    traj.f_add_parameter(
        "data.seed", np.int64(1), "Random seed"
    )
    traj.f_add_parameter(
        "data.alpha_mean", np.float64(-1.85), "Mean of the heterogeneity parameter"
    )

    # parameters (model)
    traj.f_add_parameter(
        "model.K", np.int64(3), "Number of latent components in the model"
    )
    traj.f_add_parameter(
        "model.adj_model", "Logistic", "Adjacency model"
    )
    traj.f_add_parameter(
        "model.bin_model", "Logistic", "Binary covariate model"
    )

    # parameters (fit)
    traj.f_add_parameter(
        "fit.n_iter", np.int64(10), "Number of VEM iterations"
    )
    traj.f_add_parameter(
        "fit.n_vmp", np.int64(10), "Number of VMP iterations per E-step"
    )
    traj.f_add_parameter(
        "fit.n_gd", np.int64(10), "Number of GD iterations per M-step"
    )
    traj.f_add_parameter(
        "fit.step_size", np.float64(0.01), "GD Step size"
    )

    # experiment
    explore_dict = {
        "data.missing_rate": np.arange(0.05, 0.5, 0.25),
        "data.p_cts": np.array([10, 100, 1000]),
        "data.seed": np.arange(0, 2, 1)
    }
    experiment = cartesian_product(explore_dict, ('data.missing_rate', "data.p_cts", "data.seed"))
    traj.f_explore(experiment)

    env.add_postprocessing(post_processing)
    env.run(run)
    env.disable_logging()


if __name__ == "__main__":
    main()