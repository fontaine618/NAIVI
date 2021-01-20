import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from facebook.data import get_centers, get_data

# PATH = "/home/simon/Documents/NNVI/facebook/data/raw/"
# centers = get_centers(PATH)
# node = centers[9]

plt.style.use("seaborn")

DATA_PATH = "/home/simon/Documents/NNVI/facebook/data/raw/"
FIG_PATH = "/home/simon/Documents/NNVI/facebook/figs/"

centers = get_centers(DATA_PATH)

node = centers[0]

i0, i1, A, _, X_bin, nodes = get_data(DATA_PATH, node, True)

A_df = pd.DataFrame({0: i0, 1: i1, "A": A.int().view(-1).numpy()})
A_df = A_df.loc[A_df["A"] == 1].drop(columns=["A"])

graph = nx.Graph()
for _, (i0, i1) in A_df.iterrows():
    graph.add_edge(i0, i1)
node_id = nodes[node]


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

# Ego network
pos = nx.drawing.spring_layout(graph)
nx.draw(graph, pos=pos, ax=ax1,
        node_color="#52854C", with_labels=False,
        node_size=10, width=0.05, alpha=0.3)
nx.draw_networkx_nodes(graph, pos=pos, ax=ax1, node_size=10,
        node_color="#52854C", with_labels=False, width=0.05)
nx.draw_networkx_nodes(graph, pos=pos, ax=ax1, node_size=50, nodelist=[node_id],
        node_color="#D16103", with_labels=False, width=0.05)
ax1.set_title("Ego network (density: {:.2f} %)".format(A.mean() *100))

# Covariate matrix
ax2.imshow(1.-X_bin)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.grid(None)
ax2.set_title("Node features")
ax2.set_xlabel("$p$={}".format(X_bin.size(1)))
ax2.set_ylabel("$N$={}".format(X_bin.size(0)))

# Covariate mean
ax3.hist(X_bin.mean(0).numpy(), bins=50)
ax3.set_title("Feature mean histogram")
ax3.set_xlabel("Feature mean")
ax3.set_ylabel("Frequency")
ax3.set_xlim(0, 1)

fig.tight_layout()
fig.savefig(FIG_PATH + "ego_network.pdf")