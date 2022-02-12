from pypet import Trajectory
import re, os
import pandas as pd


def load_results(name, path="/home/simon/Documents/NAIVI/sims_final/results/"):
	files = [x for x in os.listdir(path) if re.match(f"{name}.*\.hdf5", x)]
	results = {}
	parameters = {}
	for file in files:
		traj = Trajectory(file[:-5], add_time=False)
		traj.f_load(filename=path + file, load_results=0)
		traj.v_auto_load = True
		results[file] = traj.res.summary.results.df
		parameters[file] = traj.res.summary.parameters.df
	results = pd.concat(results.values(), ignore_index=True)
	parameters = pd.concat(parameters.values(), ignore_index=True)
	return results, parameters