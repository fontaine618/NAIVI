import pandas as pd
from pypet import Trajectory, ParameterGroup, Parameter, NNGroupNode
from typing import Any


def gather_results_to_DataFrame(traj: Trajectory) -> pd.DataFrame:
    traj.f_load(load_results=2, force=True)
    run_ids = list(traj.f_iter_runs())
    results_dict = dict()
    for run_id in run_ids:
        try:
            traj.f_set_crun(run_id)
            results_dict[run_id] = flatten_group_to_dict(traj.res.crun)
        except AttributeError:
            continue
    return pd.DataFrame(results_dict).T


def gather_parameters_to_DataFrame(traj: Trajectory) -> pd.DataFrame:
    run_ids = list(traj.f_iter_runs())
    parameters_dict = dict()
    for run_id in run_ids:
        try:
            traj.f_set_crun(run_id)
            parameters_dict[run_id] = flatten_group_to_dict(traj.par)
        except AttributeError:
            continue
    return pd.DataFrame(parameters_dict).T


def flatten_group_to_dict(group: NNGroupNode) -> dict[str, Any]:
    out = dict()
    for value in group:
        if isinstance(value, NNGroupNode):
            subgroup = flatten_group_to_dict(value)
            subgroup = {f"{value.v_name}.{k}": v for k, v in subgroup.items()}
            out.update(subgroup)
        else:
            out[value.v_name] = value.data
    return out