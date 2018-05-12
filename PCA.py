def plotpcascore(self, groupcol, colormap="gnuplot", bgcolor="lightgrey",
                     lcolor="black", lwidth=1, lalpha=0.50, palpha=0.50):
        """
        groupcol: the columns containing the group for each datapoint
        colormap: any (continuous) matplotlib colormap
        lwidth: width of the centerlines
        lalpha: alpha of the centerlines
        palpha: alpha of the data points
        """

        if "PC1" not in self.df or "PC2" not in self.df:
            self.addpca(2)

        groups = self.df[groupcol].unique()

        # Create a new supplot
        _, ax1 = plt.subplots(1, 1, figsize=self.plotsize)

        # Create a colorpalette
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

        # Draw centerlines and background
        ax1.axvline(0, linewidth=lwidth, color=lcolor, alpha=lalpha)
        ax1.axhline(0, linewidth=lwidth, color=lcolor, alpha=lalpha)
        ax1.set_facecolor(bgcolor)

        plt.show()

    def plotpcaloadings(self):
        # Create a new subplot
        _, ax2 = plt.subplots(1, 1, figsize=self.plotsize)
        ax2.scatter(self.eigenv[0], self.eigenv[1], color="black", s=2)

        # Label the eigenvectors (PC's)
        for i, txt in enumerate(self.colnames):
            # use horalign and veralign to place text in more readable places
            horalign = "right"
            veralign = "top"

            if self.eigenv[0][i] >= 0:
                horalign = "left"

            if self.eigenv[1][i] >= 0:
                veralign = "bottom"

            # Draw labels
            ax2.annotate(i + 1,
                         xy=(self.eigenv[0][i], self.eigenv[1][i]),
                         horizontalalignment=horalign,
                         verticalalignment=veralign)
            # Draw arrows
            ax2.annotate("",
                         xy=(self.eigenv[0][i], self.eigenv[1][i]),
                         xytext=(0, 0),
                         arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", facecolor="black"),
                         color="blue")

        # Draw centerlines
        ax2.axvline(0, linewidth=1, color="black", alpha=0.50)
        ax2.axhline(0, linewidth=1, color="black", alpha=0.50)

        plt.show()
