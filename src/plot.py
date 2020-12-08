import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class plot:

    def __init__(self, variable, df=None, terms=None, forest=None,
                 forest_feat=None):
        self.variable = variable
        self.terms = terms
        self.df = df
        self.forest = forest
        self.forest_feat = forest_feat

    def plot_importances(self):
        """ Plots importance ranking of top features in Random Forest model
        """
        n = 10
        importances = self.forest.feature_importances_[:n]
        std = np.std([tree.feature_importances_ for
                     tree in self.forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        features = list(self.forest_feat.columns[indices])

        # Print the feature ranking
        print(f"\n{n}. Feature ranking:")

        for f in range(n):
            print("%d. %s (%f)" %
                  (f + 1, features[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        _, ax = plt.subplots(figsize=(10, 15))

        ax.bar(range(n), importances[indices], yerr=std[indices],
               color="palevioletred", align="center")
        ax.set_xticks(range(n))
        ax.set_xticklabels(features, rotation=90)
        ax.set_xlim([-1, n])
        ax.set_xlabel("Importance")
        plt.xticks(rotation=30, ha='right')
        ax.set_title("Feature Importances")
        plt.show()

    def plot_variable_hist(self, colors='Set3', bin_size=32):
        """Takes in specifications and produces a graph
        Args:
            colors (str, optional): colors to be assigned to "pallete"
                feature inseaborn. Defaults to 'Set3'.
            bin_size (int, optional): Histogram bin size. Defaults to 32.
        """
        ax = sns.histplot(x=self.variable, data=self.df,
                          palette=colors, bins=bin_size)
        ax.set_xlabel(self.terms['xlabel'])
        ax.set_ylabel(self.terms['ylabel'])
        if self.variable == 'RIDRETH3':
            ax.set_xlim([0, 10])
            ax.set_xticks([0, .5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                          4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
                          9.0, 9.5, 10.0])
        ax.set_title(self.terms['title'])
        plt.show()

    def snscountplot(self, colors="Set3"):
        """Creats a histogram-like barchart that sums
            categories in a given column
        Args:
            colors (str, optional): [description]. Defaults to "Set3".
        """
        ax = sns.countplot(x=self.variable, data=self.df, palette=colors,
                           order=self.df[self.variable].value_counts().index)
        ax.set_xlabel(self.terms['xlabel'])
        ax.set_ylabel(self.terms['ylabel'])
        ax.set_title(self.terms['title'])
        if self.variable == 'RIDRETH3':
            plt.xticks(rotation=30, ha='right')  # for ethincity plot
        plt.show()
