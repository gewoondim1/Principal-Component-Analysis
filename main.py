from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


filelocation = "https://www.dropbox.com/s/4xomtdd3tee0efg/Senseo%20koffie.csv?raw=1"

df = read_csv(filelocation, delimiter=";")
print(list(df))  # columns

x = np.array(df)[:, 4:].astype(np.float32)

"""
_, ax = plt.subplots()
ax.boxplot(x.transpose().tolist())
plt.show()
"""

# Normalize the data
xc = x - x.mean(axis=0)
xcs = xc / xc.std(axis=0)

covMatrix = np.dot(xcs.transpose(), xcs) / (len(xcs)-1)
lambdaCov, pCov = [list(e) for e in np.linalg.eig(covMatrix)]

lambdaCovSorted = sorted(lambdaCov)
indices = (lambdaCov.index(i) for i in lambdaCovSorted[:2])
df["PC1"], df["PC2"] = (np.dot(xcs, pCov[i]) for i in indices)


plt.figure(figsize=(5, 5))
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']

# loop through labels and plot each cluster
for i, label in enumerate(df["Blend"].unique()):
    # add data points 
    plt.scatter(x=df.loc[df["Blend"] == label, "PC1"],
                y=df.loc[df["Blend"] == label, "PC2"],
                color=customPalette[i],
                alpha=0.50)

    # add label
    plt.annotate(label,
                 df.loc[df["Blend"] == label, ["PC1", "PC2"]].mean(),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=15, weight='bold',
                 color=customPalette[i])

plt.axvline(0, linewidth=1, color="black", alpha=0.50)
plt.axhline(0, linewidth=1, color="black", alpha=0.50)
plt.show()
