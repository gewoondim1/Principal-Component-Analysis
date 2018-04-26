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

covmatrix = np.dot(xcs.transpose(), xcs) / (len(xcs)-1)
lambdacov, pcov = [list(e) for e in np.linalg.eig(covmatrix)]

lambdacovsorted = sorted(lambdacov)
indices = (lambdacov.index(i) for i in lambdacovsorted[:2])
df["PC1"], df["PC2"] = (np.dot(xcs, pcov[i]) for i in indices)

sns.lmplot("PC1", "PC2", data=df, hue="Blend", fit_reg=False)
plt.show()
