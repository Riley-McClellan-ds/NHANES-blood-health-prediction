import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from clean_data import datapipeline
from matplotlib.ticker import FormatStrFormatter

params = {'legend.fontsize': 'xx-small',
          'axes.labelsize': 'xx-large',
          'axes.titlesize':'xx-large',
          'xtick.labelsize':'xx-large',
          'ytick.labelsize':'xx-large',}
plt.rcParams.update(params)


sns.set_theme(style="darkgrid")
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 200
# plt.rcParams['font.size'] = 20
pd.set_option('display.max_columns', None)
sns.set(font_scale=1.35)


def import_rf_nhanes(testing=False):
    clean_nhanes = datapipeline()
    target, features, df = clean_nhanes.create_rf_X_y()
    # target = target['HDL_OVER_TCHOL']  # .apply(lambda x: np.log(x+1))
    if testing:
        features['random_numbers'] = np.random.randint(1, 1000000,
                                                    size=features.shape[0])
    return target, features, df


class plot:

    def __init__(self, variable, df=None, terms=None, forest=None,
                 forest_feat=None):
        self.variable = variable
        self.terms = terms
        self.df = df
        self.forest = forest
        self.forest_feat = forest_feat


    def importance_columns_renamer(self):
        """Renames columns for creation of feature importance graph

        Args:
            model_nhanes_df ([DataFrame]): the dataframe used to create
            our random forest model.

        Returns:
            [Dataframe]: [A Dataframe to be used only for graphing puropses.]
        """
        naming_columns = {
            'BMXBMI': 'BMI',
            'BMXWAIST': 'Waist size',
            "RIDAGEYR": 'Age',
            'BMXHT': 'Height',
            'BPXSY1': 'S. Blood Pressure',
            'BPXDI1': 'D. Blood Pressure',
            "SUMTEETH": "Missing Teeth",
            "DMDHHSIZ": "Household size",
            "RIDEXMON": 'Season of Data collected',
            "DMDCITZN": "Citizenship status",
            "BMXLEG": "Leg Length",
            "DR1TPFAT": "Dietary Fat",
            "DR1TCHOL": "Dietary Chol.",
            "BMXARMC": "Arm Circ.",
            "WHD140": "Max Weight",
            "age_max_weight": "Age Max Weight",
            "BMXARML": "Arm Length",
            "RIAGENDR": "Gender",
            "PAD680": "Time Sedentary",
            "PAQ635": "Walking/Cycling",
            "PAQ620": "Moderate W. Activity",
            "PAQ605": "Vigorous W. Activity",
            "PAQ665": "Moderate R. Activity",
            "PAQ650": "Vigorous R. Activity",
            'FITNESS_SCORE_MIN': "Fitness Score"
        }
        naming_columns2 = {
            'BMXBMI': ' ',
            'BMXWAIST': ' ',
            "RIDAGEYR": ' ',
            'BMXHT': ' ',
            'BPXSY1': ' ',
            'BPXDI1': ' ',
            "SUMTEETH": " ",
            "DMDHHSIZ": " ",
            "RIDEXMON": ' ',
            "DMDCITZN": " ",
            "BMXLEG": " ",
            "DR1TPFAT": " ",
            "DR1TCHOL": " ",
            "BMXARMC": " ",
            "WHD140": " ",
            "age_max_weight": " ",
            "BMXARML": " ",
            "RIAGENDR": " ",
            "PAD680": "",
            "PAQ635": "Walking/Cycling",
            "PAQ620": "Mod. W. Activity",
            "PAQ605": "Vig. W. Activity",
            "PAQ665": "Mod. R. Activity",
            "PAQ650": "Vig. R. Activity",
            'FITNESS_SCORE_MIN': "Fitness Score"
        }
        naming_columns3 = {
            'BMXBMI': ' ',
            'BMXWAIST': 'Waist Size',
            "RIDAGEYR": ' ',
            'BMXHT': ' ',
            'BPXSY1': ' ',
            'BPXDI1': ' ',
            "SUMTEETH": " ",
            "DMDHHSIZ": " ",
            "RIDEXMON": ' ',
            "DMDCITZN": " ",
            "BMXLEG": " ",
            "DR1TPFAT": " ",
            "DR1TCHOL": " ",
            "BMXARMC": " ",
            "WHD140": " ",
            "age_max_weight": " ",
            "BMXARML": " ",
            "RIAGENDR": " ",
            "PAD680": "",
            "PAQ635": "",
            "PAQ620": "",
            "PAQ605": "",
            "PAQ665": "",
            "PAQ650": "",
            'FITNESS_SCORE_MIN': ""
        }
        self.forest_feat = self.forest_feat.rename(columns=naming_columns2)
        return self.forest_feat

    def plot_importances(self, n=10):
        """ Plots importance ranking of top features in Random Forest model
        """
        
        self.importance_columns_renamer()
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
        og = "#FF9F26"
        bl = "#0D8295"
        # Plot the feature importances of the forest
        _, ax = plt.subplots(figsize=(10, 15))
        colort = ["#FF9F26", "#FF9F26", "#FF9F26", "#FF9F26",
                  "#FF9F26", "#FF9F26", "#FF9F26", '#0D8295',
                  "#FF9F26", "#FF9F26", "#FF9F26", "#FF9F26",
                  "#FF9F26", "#FF9F26", "#FF9F26", "#FF9F26"]
        colors = ["#FF9F26", "#FF9F26", "#FF9F26", "#FF9F26",
                  "#FF9F26", "#FF9F26", "#FF9F26", "#FF9F26",
                  "#FF9F26", "#FF9F26", "#FF9F26", "#FF9F26",
                  "#FF9F26", "#FF9F26", "#FF9F26", "#FF9F26",
                  '#0D8295', '#0D8295', '#0D8295',
                  '#0D8295', '#0D8295']
        colort = [og, og, og, og, og, og, og, bl, og, bl, og, bl, og,
                  og, og, og]
        ax.bar(range(n), importances[indices], yerr=std[indices],
               color=colors, align="center")
        ax.set_xticks(range(n))
        ax.set_xticklabels(features, rotation=90)
        ax.set_xlim([-1, n])
        # ax.set_xlabel("Importance")
        plt.xticks(rotation=30, ha='right')
        ax.set_title("Feature Importances")
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
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

    def scatterplot(self, variable2):
        sns.scatterplot(data=self.df, x=self.variable, y=variable2)
        plt.show()

    def correct_target_bins(self):
        """Creates fake values that cause the bins of the
        cholesterol graph to line up with 0 and 10
        Returns:
            (Pandas DataFrame): temporary version of datframe
            for graphing purposes only
        """
        self.df = self.df[self.df['HDL_OVER_TCHOL'] <= 10]
        print(self.df['HDL_OVER_TCHOL'].max())
        self.df.loc[:1, 'HDL_OVER_TCHOL'] = 0.0
        self.df.loc[1:2, 'HDL_OVER_TCHOL'] = 0.6
        self.df.loc[2:3, 'HDL_OVER_TCHOL'] = 10.0
        return self.df

    def create_target(self):
        self.df['HDL_OVER_TCHOL'] = self.df['LBDHDD']/self.df['LBXTC']
        self.df['Cholesterol Ratio'] = self.df['LBXTC']/self.df['LBDHDD']
        return self.df

    def make_age(self):
        
        self.df['Age Group'] = '20-29'
        self.df.loc[self.df['RIDAGEYR'] > 29.0, 'Age Group'] = '30-39'
        self.df.loc[self.df['RIDAGEYR'] > 39.0, 'Age Group'] = '40-49'
        self.df.loc[self.df['RIDAGEYR'] > 49.0, 'Age Group'] = '50-59'
        self.df.loc[self.df['RIDAGEYR'] > 59.0, 'Age Group'] = '60-69'
        self.df.loc[self.df['RIDAGEYR'] > 69.0, 'Age Group'] = '70-80'
        return self.df

    def sns_violin(self):
        self.make_age()
        self.create_target()
        ax = sns.violinplot(x="Age Group", y="Cholesterol Ratio", hue="Gender",
                    data=self.df, palette=['lightpink', 'lightskyblue'], 
                    split=True, scale="count")
        plt.title('Participant Age VS Cholesterol')
        plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.6, 1.0))
        plt.show()

if __name__ == '__main__':
    target, features, df = import_rf_nhanes(testing=False)
    features['age_max_weight'] = features['RIDAGEYR'] - features['WHQ150']
    # features['max_weight_by_age'] *= features['WHD140']
    features =  features.drop('WHQ150', axis=1)

    #  Adjust labels as needed for plotting
    labels = {
        'title': '1',
        'xlabel': '2',
        'ylabel': '3'
    }
    df['Gender'] = 'Male'
    df.loc[df['RIAGENDR'] == 2.0, 'Gender'] = 'Female'
    df.loc[0, 'RIDAGEYR'] = 21.0
    df.loc[3, 'RIDAGEYR'] = 31.0
    df.loc[8, 'RIDAGEYR'] = 31.0
    df.loc[4, 'RIDAGEYR'] = 41.0
    df.loc[5, 'RIDAGEYR'] = 51.0
    df.loc[2, 'RIDAGEYR'] = 61.0
    line = plot('LBDHDD', df=df, terms=labels)
    line.sns_violin()
    pass
