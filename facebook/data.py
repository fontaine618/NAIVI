import pandas as pd
import numpy as np
import torch
import os


def get_centers(PATH):
    files = os.listdir(PATH)
    nodes = set()
    for file in files:
        end = file.find(".")
        node = int(file[:end])
        nodes.add(node)
    return sorted(list(nodes))


def get_data(PATH, node):
    # import files
    edges = pd.read_csv("{}{}.edges".format(PATH, node), header=None, sep=" ")
    features = pd.read_csv("{}{}.feat".format(PATH, node),
                           header=None, sep=" ", index_col=0)
    # get all nodes
    nodes = set(edges.to_numpy().reshape(-1))
    nodes.update(features.index)
    # recreate the adjacency matrix
    edges_sorted = edges.copy()
    edges_sorted[0] = np.where(edges[0] > edges[1], edges[0], edges[1])
    edges_sorted[1] = np.where(edges[0] > edges[1], edges[1], edges[0])
    edges_sorted.drop_duplicates(inplace=True)
    # node id to integer
    nodes_dict = {node: i for i, node in enumerate(sorted(nodes))}
    edges_sorted.replace(nodes_dict, inplace=True)
    edges_sorted["A"] = 1.
    edges_sorted.set_index([0, 1], inplace=True)
    N = len(nodes)
    i = torch.tril_indices(N, N, -1).t()
    i0 = i[:, 0]
    i1 = i[:, 1]
    i = [tuple(ii) for ii in i.numpy()]
    edges_tmp = pd.DataFrame(data=np.zeros((len(i))), index=pd.MultiIndex.from_tuples(i, names=[0, 1]))
    edges_tmp.columns = ["A"]
    edges_tmp.update(edges_sorted)
    # add missing rows to features
    for i in nodes:
        if i not in features.index:
            features.loc[i] = np.nan
    features.index = [nodes_dict[i] for i in features.index]
    features.sort_index(inplace=True)
    # drop columns with rare
    which = features.var() > 0.01
    features = features.loc[:, which]
    # output
    A = torch.tensor(edges_tmp["A"].to_numpy()).view((-1, 1))
    X_cts = None
    X_bin = torch.tensor(features.to_numpy()).double()
    return i0, i1, A, X_cts, X_bin


