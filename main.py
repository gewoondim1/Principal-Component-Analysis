from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

# import seaborn as sns


class PCA:
    def __init__(self, dataframe, columns=None, startvar=0, endvar=0):
        self.df = dataframe

        if columns is []:
            raise KeyError("Columns can not be an empty list")

        try:
            if columns:
                self.x = np.array(dataframe)[:, columns].astype(np.float32)
                self.colnames = [list(dataframe)[i] for i in columns]
            elif not endvar:
                self.x = np.array(dataframe)[:, startvar:].astype(np.float32)
                self.colnames = list(dataframe)[startvar:]
            else:
                self.x = np.array(dataframe)[:, startvar:endvar].astype(np.float32)
                self.colnames = list(dataframe)[startvar:endvar]
        except ValueError:
            raise ValueError("All data has to be numerical or strings containing only numbers.")

        self.eigenv = []

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
            self.eigenv.append(pcov[eig])

    def plotpcascore(self, groupcol, colormap="gnuplot", bgcolor="lightgrey",
                lcolor="black", lwidth=1, lalpha=0.50, palpha=0.50):
        if "PC1" not in self.df or "PC2" not in self.df:
            self.addpca(2)

        groups = self.df[groupcol].unique()
        _, ax1 = plt.subplots(1, 1)

        cmap = plt.get_cmap(colormap)
        colorpalette = [cmap(c) for c in np.linspace(0, 1, len(groups))]

        # Loop through labels and plot each cluster
        for i, label in enumerate(groups):
            # Add data points
            ax1.scatter(x=self.df.loc[self.df[groupcol] == label, "PC1"],
                        y=self.df.loc[self.df[groupcol] == label, "PC2"],
                        color=colorpalette[i],
                        alpha=palpha)

            # Add labels
            ax1.annotate(label,
                         self.df.loc[self.df[groupcol] == label, ["PC1", "PC2"]].mean(),
                         horizontalalignment="center",
                         verticalalignment="center",
                         size=15, weight="bold",
                         color=colorpalette[i])

        ax1.axvline(0, linewidth=lwidth, color=lcolor, alpha=lalpha)
        ax1.axhline(0, linewidth=lwidth, color=lcolor, alpha=lalpha)
        ax1.set_facecolor(bgcolor)

        plt.show()

    def plotpcaloadings(self):
        _, ax2 = plt.subplots(1, 1)
        ax2.scatter(self.eigenv[0], self.eigenv[1], color="black", s=2)

        for i, txt in enumerate(self.colnames):
            horalign = "right"
            veralign = "top"

            if self.eigenv[0][i] >= 0:
                horalign = "left"

            if self.eigenv[1][i] >= 0:
                veralign = "bottom"

            ax2.annotate(i+1,
                         xy=(self.eigenv[0][i], self.eigenv[1][i]),
                         horizontalalignment=horalign,
                         verticalalignment=veralign)
            ax2.annotate("",
                         xy=(self.eigenv[0][i], self.eigenv[1][i]),
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor="black"),
                         color="blue")

        ax2.axvline(0, linewidth=1, color="black", alpha=0.50)
        ax2.axhline(0, linewidth=1, color="black", alpha=0.50)

        plt.show()


filelocation = "https://www.dropbox.com/s/4xomtdd3tee0efg/Senseo%20koffie.csv?raw=1"

df = read_csv(filelocation, delimiter=";")
# print(list(df))  # columns

pca = PCA(df, startvar=4)
pca.normalize()
pca.addpca(2)
# pca.plotpcascore("Blend")
pca.plotpcaloadings()
