import gc

import torch
from pypet import Environment, cartesian_product, Trajectory, Parameter, ParameterGroup
from .data import Dataset
from .method import Method
from .results import Results


def run(traj: Trajectory):
    # do some GC here since pypet won't do it well for torch.
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    # get data instance (this could be loaded data or synthetic data)
    data: Dataset = Dataset.from_parameters(traj.data)
    # get method instance
    method: Method = Method.from_parameters(traj.method, traj.model)
    # run method on data and get results
    results: Results = method.fit(data, traj.fit)
    # save results
    results_dict = results.to_dict()
    for k, v in results_dict.items():
        traj.f_add_result(f"$.{k}", v)
    # do some GC here since pypet won't do it well for torch.
    del data, method, results, results_dict
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()