import pandas as pd


def post_processing(traj, result_list):

    runs = [res[0] for res in result_list]

    results = pd.concat([pd.DataFrame(result_list[i][1], index=[i]) for i in runs])

    parameters = parameter_df(runs, traj)

    traj.f_add_result("summary.results", df=results)
    traj.f_add_result("summary.parameters", df=parameters)


def parameter_df(runs, traj):
    parameters = dict()
    for group_name, group in traj.par.f_get_groups().items():
        for par_name, par in group.f_get_children().items():
            try:
                values = par.f_get_range()
            except TypeError:  # patch for default
                values = [par.f_get_default()] * len(runs)
            parameters[(group_name, par_name)] = values
    df = pd.DataFrame(parameters, index=runs)
    return df
