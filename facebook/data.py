import pandas as pd
import os

PATH = "/home/simon/Documents/NNVI/facebook/data/raw/"


def get_centers(PATH):
    files = os.listdir(PATH)
    nodes = set()
    for file in files:
        end = file.find(".")
        node = int(file[:end])
        nodes.add(node)
    return sorted(list(nodes))


def get_data(PATH, node):
    edges = pd.read_csv("{}{}.edges".format(PATH, node), header=None, sep=" ")
    nodes = set(edges.to_numpy().reshape(-1))
    nodes_dict = {node: i for i, node in enumerate(nodes)}
    edges.replace(nodes_dict, inplace=True)
    features = pd.read_csv("{}{}.feat".format(PATH, node),
                           header=None, sep=" ", index_col=0)
    features.drop(index=[i for i in features.index if i not in nodes], inplace=True)
    features.index = [nodes_dict[i] for i in features.index]
    features.sort_index(inplace=True)


nodes = get_centers(PATH)
node = nodes[3]