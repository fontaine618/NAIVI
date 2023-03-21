import gc
import torch
from pypet import Environment, cartesian_product, Trajectory, Parameter, ParameterGroup
from .data import Dataset
from .method import Method
from .results import Results


def run(traj: Trajectory):
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    # do some GC here since pypet won't do it well for torch.
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    print_run_summary(traj)
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


def print_run_summary(traj: Trajectory):
    out = f"""
================================================================================
RUN SETTINGS
--------------------------------------------------------------------------------\n"""
    for v in traj.par.f_iter_leaves():
        if isinstance(v, Parameter):
            out += f"{v.v_full_name:50s} : {v.data}\n"
    out += f"================================================================================"
    print(out)