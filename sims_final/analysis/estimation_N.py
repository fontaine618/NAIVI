import pandas as pd
import numpy as np
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn")
from pypet import Trajectory
import itertools as it
from matplotlib.lines import Line2D

RESULTS_PATH = "/home/simon/Documents/NAIVI/sims_final/results/"
FIGS_PATH = "/home/simon/Documents/NAIVI/sims_final/figs/"
EXP_NAME = "estimation_N"

traj = Trajectory(EXP_NAME, add_time=False)
traj.f_load(filename=RESULTS_PATH + EXP_NAME + "/gpu_seed0.hdf5", force=True)
traj.v_auto_load = True

traj.par