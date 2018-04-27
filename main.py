from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns


class PCA:
    def __init__(self, dataframe, columns=None, startvar=0, endvar=0):
        self.df = dataframe

        if columns:
            self.x = np.array(dataframe)[:, columns].astype(np.float32)
        elif not endvar:
            self.x = np.array(dataframe)[:, startvar:].astype(np.float32)
        else:
            self.x = np.array(dataframe)[:, startvar:endvar].astype(np.float32)

    def normalize(self, normalize=True):
        self.x -= self.x.mean(axis=0)

        if normalize:
            self.x /= self.x.std(axis=0)

    def addpca(self, n=2):
        covmatrix = np.dot(self.x.transpose(), self.x) / (len(self.x) - 1)
        lambdacov, pcov = [list(e) for e in np.linalg.eig(covmatrix)]

        lambdacovsorted = sorted(lambdacov)
        indices = (lambdacov.index(i) for i in lambdacovsorted[:n])

        for i, eig in enumerate(indices, 1):
            self.df["PC" + str(i)] = np.dot(self.x, pcov[eig])


filelocation = "https://www.dropbox.com/s/4xomtdd3tee0efg/Senseo%20koffie.csv?raw=1"

df = read_csv(filelocation, delimiter=";")
print(list(df))  # columns

x = PCA(df, startvar=4)
x.normalize()
x.addpca()

plt.figure(figsize=(5, 5))
customPalette = ['#630C3A', '#39C8C6', '#D3500C', '#FFB139']

# loop through labels and plot each cluster
for i, label in enumerate(x.df["Blend"].unique()):
    # add data points 
    plt.scatter(x=x.df.loc[x.df["Blend"] == label, "PC1"],
                y=x.df.loc[x.df["Blend"] == label, "PC2"],
                color=customPalette[i],
                alpha=0.50)

    # add label
    plt.annotate(label,
                 x.df.loc[x.df["Blend"] == label, ["PC1", "PC2"]].mean(),
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=15, weight='bold',
                 color=customPalette[i])

plt.axvline(0, linewidth=1, color="black", alpha=0.50)
plt.axhline(0, linewidth=1, color="black", alpha=0.50)
plt.show()
