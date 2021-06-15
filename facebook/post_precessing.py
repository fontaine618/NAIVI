import pandas as pd


def post_processing(traj, result_list):

    run_idx = [res[0] for res in result_list]

    df_in = pd.DataFrame([res[1][0] for res in result_list])
    df_in.index = run_idx
    df_in.columns = [
        "center",
        "seed", "N", "p_cts", "p_bin",
        "K", "missing_rate", "alpha_mean",
        "K_model", "algo"
    ]
    print(df_in)

    df_out = pd.DataFrame([res[1][1] for res in result_list])
    df_out.index = run_idx
    print(df_out)
    df_out.columns = [
        "n_iter", "grad_norm",
        "train_loss", "train_mse", "train_auroc",
        "dist_inv", "dist_proj",
        "test_loss", "test_mse", "test_auroc",
        "aic", "bic",
        "density", "time"
    ]

    df = pd.concat([df_in, df_out], axis=1)

    print(df)
    traj.f_add_result("data_frame", df, "Summary across replications")
    df.to_csv(traj.par.path + "/results/summary.csv")