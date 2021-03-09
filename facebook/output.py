import numpy as np
import pandas as pd
from facebook.data import get_data, get_centers

PATH = "//facebook/data/raw/"
# PATH = "/home/simfont/NAIVI/facebook/data/raw/"

centers = get_centers(PATH)

out = []
for node in centers:
    i0, i1, A, X_cts, X_bin = get_data(PATH, node)
    out.append([node, X_bin.size(0), X_bin.size(1), A.mean().item()])

table = pd.DataFrame(data=np.array(out))
table.columns = ["Center", "N", "p", "Density (%)"]
table["Center"] = table["Center"].astype(int)
table["N"] = table["N"].astype(int)
table["p"] = table["p"].astype(int)
table["Density (%)"] = table["Density (%)"].apply(lambda x: "{:.1f}".format(x*100))
table.set_index("Center", inplace=True)

print(table.to_latex())
print(table.transpose().to_latex())