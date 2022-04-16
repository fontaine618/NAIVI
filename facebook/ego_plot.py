import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
from facebook.data import get_centers, get_data


# all ego networks
PATH = "./facebook/data/raw/"
centers = get_centers(PATH)
out = []
for node in centers:
    i0, i1, A, X_cts, X_bin = get_data(PATH, node)
    out.append([node, X_bin.size(0), X_bin.size(1), A.mean().item()])

table = pd.DataFrame(data=np.array(out))
table.columns = ["Center", "N", "p", "density"]
table["Center"] = table["Center"].astype(int)
table["N"] = table["N"].astype(int)
table["p"] = table["p"].astype(int)
table.set_index("Center", inplace=True)

# rest

plt.style.use("seaborn-whitegrid")

DATA_PATH = "./facebook/data/raw/"
FIG_PATH = "./facebook/figs/"

centers = get_centers(DATA_PATH)

node = centers[5]

# graph without ego
i0, i1, A, _, X_bin, nodes = get_data(DATA_PATH, node, True, False)
A_df = pd.DataFrame({0: i0, 1: i1, "A": A.int().view(-1).numpy()})
A_df = A_df.loc[A_df["A"] == 1].drop(columns=["A"])
graph = nx.Graph()
for _, (i0, i1) in A_df.iterrows():
    graph.add_edge(i0, i1)

# graph with ego
i0, i1, A_ego, _, _, nodes = get_data(DATA_PATH, node, True, True)
A_df = pd.DataFrame({0: i0, 1: i1, "A": A_ego.int().view(-1).numpy()})
A_df = A_df.loc[A_df["A"] == 1].drop(columns=["A"])
ego_graph = nx.Graph()
for _, (i0, i1) in A_df.iterrows():
    ego_graph.add_edge(i0, i1)
node_id = nodes[node]

# corr mat
corrmat = pd.DataFrame(X_bin.numpy()).corr()
cg = sns.clustermap(corrmat.abs(), cmap="RdBu")
order = cg.dendrogram_col.dendrogram["leaves"]
corrmat = corrmat.to_numpy()
corrmat = corrmat[:, order]
corrmat = corrmat[order, :]


# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(11.5, 2.5))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7, 5.45))
# Ego network
pos = nx.drawing.spring_layout(ego_graph)
nx.draw(ego_graph, pos=pos, ax=ax1,
        node_color="#52854C", with_labels=False,
        node_size=10, width=0.05, alpha=0.0)
nx.draw_networkx_edges(ego_graph, pos=pos, ax=ax1,
                       edgelist=[e for e in ego_graph.edges() if node_id not in e],
                       width=0.05, alpha=0.2)
nx.draw_networkx_nodes(ego_graph, pos=pos, ax=ax1, node_size=10,
        node_color="#4c72b0", with_labels=False, width=0.05)
# nx.draw_networkx_nodes(ego_graph, pos=pos, ax=ax1, node_size=50, nodelist=[node_id],
#         node_color="#c44e52", with_labels=False, width=0.05)
ax1.set_title("Ego network (#{})".format(node))
# correlation

ax2 = sns.heatmap(corrmat, ax=ax2, cmap="RdBu", vmin=-1., vmax=1.)
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
ax2.set_title("Correlation matrix (#{})".format(node))
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_aspect('equal')
# Covariate mean
ax3.hist(X_bin.mean(0).numpy(), bins=20, color="#4c72b0")
ax3.set_title("Attribute proportion (#{})".format(node))
ax3.set_xlabel("Proportion")
ax3.set_ylabel("Frequency")
ax3.set_xlim(0, 1)
#ego entworks
plot = ax4.scatter(table["N"], table["p"], c=table["density"], cmap="Blues", s=60,
                   linewidth=1, edgecolors="white")
ax4.set_xlabel("Network size")
ax4.set_ylabel("Nb. attributes")
ax4.set_xlim(0, 1200)
for center, row in table.iterrows():
    ax4.annotate(center, (row["N"]+50, row["p"]-3), fontsize=8)
plt.colorbar(plot, ax=ax4, label="Density")
ax4.set_title("All ego networks")

# ax3.patch.set_facecolor('#EEEEEE')
# ax4.patch.set_facecolor('#EEEEEE')

fig.tight_layout()
fig.savefig(FIG_PATH + "ego_network.pdf")
# fig.savefig(FIG_PATH + "ego_network_slides.pdf")