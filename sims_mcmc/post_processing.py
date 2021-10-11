import pandas as pd


def post_processing(traj, result_list):

    run_idx = [res[0] for res in result_list]

    df_in = pd.DataFrame([res[1][0] for res in result_list])
    df_in.index = run_idx
    df_in.columns = [
        "seed", "N", "p_cts", "p_bin",
        "K", "missing_mean", "alpha_mean",
        "K_model", "mnar", "algo", "mnar_sparsity", "n_sample"
    ]
    print(df_in)

    df_out = pd.DataFrame([res[1][1] for res in result_list])
    df_out.index = run_idx
    df_out.columns = [
        "DZ", "DP", "DThetaX", "time"
    ]
    print(df_out)

    df = pd.concat([df_in, df_out], axis=1)

    print(df)
    traj.f_add_result("data_frame", df, "Summary across replications")
    df.to_csv(traj.par.path + "summary.csv")