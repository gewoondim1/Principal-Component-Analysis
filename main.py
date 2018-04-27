from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt


class PCA:
    def __init__(self, dataframe, columns=None, startvar=0, endvar=0):
        self.df = dataframe

        if columns is []:
            raise KeyError("Columns can not be an empty list")

        try:
            if columns:
                self.x = np.array(dataframe)[:, columns].astype(np.float32)
            elif not endvar:
                self.x = np.array(dataframe)[:, startvar:].astype(np.float32)
            else:
                self.x = np.array(dataframe)[:, startvar:endvar].astype(np.float32)
        except ValueError:
            raise ValueError("All data has to be numerical or strings containing only numbers.")

    def normalize(self, normalize=True):
        self.x -= self.x.mean(axis=0)

        if normalize:
            self.x /= self.x.std(axis=0)

    def addpca(self, n):
        covmatrix = np.dot(self.x.transpose(), self.x) / (len(self.x) - 1)
        lambdacov, pcov = [list(e) for e in np.linalg.eig(covmatrix)]

        lambdacovsorted = sorted(lambdacov)
        indices = (lambdacov.index(i) for i in lambdacovsorted[:n])

        for i, eig in enumerate(indices, 1):
            self.df["PC" + str(i)] = np.dot(self.x, pcov[eig])

    def plotpca(self, groupcol, colormap="gnuplot", bgcolor="lightgrey", lcolor="black", lwidth=1, lalpha=0.50, palpha=0.50):
        if "PC1" not in self.df or "PC2" not in self.df:
            self.addpca(2)

        groups = self.df[groupcol].unique()
        _, ax = plt.subplots(1, 1)

        cmap = plt.get_cmap(colormap)
        colorpalette = [cmap(c) for c in np.linspace(0, 1, len(groups))]

        # Loop through labels and plot each cluster
        for i, label in enumerate(groups):
            # Add data points
            ax.scatter(x=self.df.loc[self.df[groupcol] == label, "PC1"],
                       y=self.df.loc[self.df[groupcol] == label, "PC2"],
                       color=colorpalette[i],
                       alpha=palpha)

            # Add labels
            ax.annotate(label,
                        self.df.loc[self.df[groupcol] == label, ["PC1", "PC2"]].mean(),
                        horizontalalignment="center",
                        verticalalignment="center",
                        size=15, weight='bold',
                        color=colorpalette[i])

        ax.axvline(0, linewidth=lwidth, color=lcolor, alpha=lalpha)
        ax.axhline(0, linewidth=lwidth, color=lcolor, alpha=lalpha)
        ax.set_facecolor(bgcolor)

        plt.show()


filelocation = "https://www.dropbox.com/s/4xomtdd3tee0efg/Senseo%20koffie.csv?raw=1"

df = read_csv(filelocation, delimiter=";")
# print(list(df))  # columns

pca = PCA(df, startvar=4)
pca.normalize()
pca.addpca(2)
pca.plotpca("Blend")
