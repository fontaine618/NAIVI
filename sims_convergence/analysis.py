from pypet import Trajectory
traj = Trajectory('example', add_time=False)
traj.f_load(filename="/home/simon/Documents/NAIVI/"
                     "results/example.hdf5")
traj.v_auto_load = True
traj.res.summary.results.df
traj.res.summary.hyperparameters.df[("fit", "algo")]

traj.v_idx = 2

traj.results.logs.crun.df
traj.results.diagnostics.crun.df.shape
