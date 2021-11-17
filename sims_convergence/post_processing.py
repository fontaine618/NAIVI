import pandas as pd


def post_processing(traj, result_list):

    runs = [res[0] for res in result_list]

    results = pd.concat([pd.DataFrame(result_list[i][1][0], index=[i]) for i in runs])

    logs = pd.concat([
        result_list[i][1][1].set_index(
            pd.MultiIndex.from_product([[i], result_list[i][1][1].index]))
        for i in runs])

    parameters = dict()
    for group_name, group in traj.par.f_get_groups().items():
        for par_name, par in group.f_get_children().items():
            try:
                values = par.f_get_range()
            except TypeError:
                values = [par.f_get_default()]*len(runs)
            parameters[(group_name, par_name)] = values
    parameters = pd.DataFrame(parameters, index=runs)

    traj.f_add_result("summary.logs", df=logs)
    traj.f_add_result("summary.results", df=results)
    traj.f_add_result("summary.parameters", df=parameters)



# from pypet import Trajectory
# traj = Trajectory('test_experiment', add_time=False)
# traj.f_load(filename="/home/simon/Documents/NAIVI/sims_convergence/"
#                      "results/test_experiment/test_experiment.hdf5",
#             load_results=0)
# traj.v_auto_load = True
# traj.res.summary.results.df