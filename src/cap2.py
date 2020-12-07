import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import (
    cluster, datasets,
    decomposition, ensemble, manifold,
    random_projection, preprocessing)
import missingno as msno


sns.set_theme(style="darkgrid")
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 140
pd.set_option('display.max_columns', None)


def load_NHANES_data():
    """Loads nhanes data by splitting a strings containing lists of files
    indto individual file names and then converting those files
    from XPT format into a pandas DataFrame then merging appropriate
    files into output Datafram

    Extra files are left in code for ease of addition into main DataFrame

    Returns:
        [pandas DataFrame]: A DataFrame containing complete column list from each
        file selected for use.
    """    

    lab = "HSCRP_J.XPT HDL_J.XPT TCHOL_J.XPT FASTQX_J.XPT"
    more_lab = "ALB_CR_J.XPT GLU_J.XPT CRCO_J.XPT FOLFMS_J.XPT \
    HEPB_S_J.XPT  IHGEM_J.XPT   UCM_J.XPT  VIC_J.XPT BIOPRO_J.XPT \
    GHB_J.XPT  HEPC_J.XPT  INS_J.XPT UCPREG_J.XPT  VOCWB_J.XPT CBC_J.XPT \
    FERTIN_J.XPT   HEPE_J.XPT  PBCD_J.XPT    UHG_J.XPT CMV_J.XPT \
    FETIB_J.XPT   HEPA_J.XPT    HIV_J.XPT   UIO_J.XPT COT_J.XPT FOLATE_J.XPT  \
        HEPBD_J.XPT   HSCRP_J.XPT   UCFLOW_J.XPT  UNI_J.XPT"
    questionnaire = "DUQ_J.XPT PAQ_J.XPT SMQ_J.XPT SMQFAM_J.XPT"  
    # ADDITIONASL QUESTIONNAIRE: ACQ_J.XPT  DEQ_J.XPT  ECQ_J.XPT  IMQ_J.XPT
    # OSQ_J.XPT RHQ_J.XPT  SMQFAM_J.XPT  WHQMEC_J.XPT AUQ_J.XPT  DIQ_J.XPT
    # HEQ_J.XPT  KIQ_U_J.XPT  PAQ_J.XPT  RXQASA_J.XPT  SMQ_J.XPT BPQ_J.XPT
    # DLQ_J.XPT  HIQ_J.XPT  MCQ_J.XPT  PAQY_J.XPT
    # RXQ_DRUG.xpt  SMQRTU_J.XPT CDQ_J.XPT  DPQ_J.XPT
    # HSQ_J.XPT  OCQ_J.XPT   PFQ_J.XPT   RXQ_RX_J.XPT
    # SMQSHS_J.XPT DBQ_J.XPT  DUQ_J.XPT  HUQ_J.XPT  OHQ_J.XPT
    # PUQMEC_J.XPT  SLQ_J.XPT  WHQ_J.XPT"
    demo = "DEMO_J.XPT"
    diet = "DR1TOT_J.XPT  DR2TOT_J.XPT  DS1TOT_J.XPT  DS2TOT_J.XPT  DSQTOT_J.XPT"
    exam = 'BMX_J.XPT BPX_J.XPT  OHXDEN_J.XPT'
    # THE FOLLOWING WERE REMOVED BECAUSE THEY ARE MORE EXPENSIVE THAN \
    # A BLOOD TEST 'OHXREF_J.XPT DXXFEM_J.XPT  DXX_J.XPT  DXXSPN_J.XPT  LUX_J.XPT '

    lst = [demo, exam, questionnaire, lab, more_lab, diet ]

    path = "/home/riley/Desktop/DSI/capstone2/"
    file_locations = ['demographics/',  'exam/',  'questionnaire_/', 'lab/', \
         'lab/morelab/', 'dietary/']

    for idx, i in enumerate(lst):
        i = i.replace("    ", " ")
        i = i.replace("   ", " ")
        i = i.replace("  ", " ")
        i = i.split(" ")
        lst[idx] = i

    df_dic = dict()

    for idx, folder in enumerate(lst):
        for file in folder:
            df_dic[file] = pd.read_sas(f'{path}{file_locations[idx]}{file}',\
                 format='xport', encoding='utf-8')

    lab_df = df_dic['HDL_J.XPT'].merge(df_dic['TCHOL_J.XPT'], \
        on='SEQN', how='outer').merge(df_dic['FASTQX_J.XPT'], on='SEQN', how='outer')


    x_data = lst[:2]

    nhanes_df = lab_df.copy()
    for idx, folder in enumerate(x_data):
        for file in folder:
            nhanes_df = nhanes_df.merge(df_dic[file],\
                 on='SEQN', how='left') #, lsuffix='_left', rsuffix='_right')

    nhanes_df = nhanes_df.merge(df_dic['DUQ_J.XPT'], on='SEQN', how='left')
    nhanes_df = nhanes_df.merge(df_dic['PAQ_J.XPT'], on='SEQN', how='left')
    nhanes_df = nhanes_df.merge(df_dic['SMQ_J.XPT'], on='SEQN', how='left')
    nhanes_df = nhanes_df.merge(df_dic['SMQFAM_J.XPT'], on='SEQN', how='left')
    
    nhanes_df = nhanes_df[nhanes_df['RIDEXAGM'].isnull()]
    return nhanes_df

nhanes_df = load_NHANES_data()


def rem_nan_cols(df):
    """Identifies columns with NaN values and returns number of NaN values
        per col
    Args:
        df (pandas DataFrame) :
    Returns:
       2 [List] : 2 lists in parralell with column name and number of NaN
       values respectively
       1 [set]: a set of columns where more than 80% of the data are NaN
    """
    nan_columns = []
    nan_nums = [] # in parrallel with nan_columns
    df_over_80_nan = set()
    for i in df.columns:
        if df[i].isnull().values.any():
            nan_columns.append(i)
    for i in df.columns:
        if df[i].isnull().sum():
            if df[i].isnull().sum() > len(df)*.8:
                df_over_80_nan.add(i)           
            nan_nums.append(df[i].isnull().sum())
    return nan_columns, nan_nums, df_over_80_nan


def check_nan_amount(df, columns):
    '''Detects and locates presence of NaN values.
    Args:
        df (pandas DataFrame):
        columns (list(STR)): List of columns to search
    Returns:
        (INT, SET) :a tuple containing a SET of indexes where there is a
        NaN value in one of the given columns as well as
        an INT representing the length of that set.
    '''
    idx = set()
    for i in columns:
        for b in df[df[i].isnull()].index.values:
            idx.add(b)
    return len(idx), idx


# target_cols = ['LBDHDD', 'LBXTC', 'LBXHSCRP'] 
# 0 is hdl #1 is total cholesterol #2 is crp


nhanes_df['HDL_OVER_TCHOL'] = nhanes_df['LBDHDD']/nhanes_df['LBXTC'] #for model
# nhanes_df['HDL_OVER_TCHOL'] = nhanes_df['LBXTC']/nhanes_df['LBDHDD'] #for plot

target_df = pd.DataFrame()
target_df['SEQN'] = nhanes_df['SEQN']
target_df['HDL_OVER_TCHOL'] = nhanes_df['HDL_OVER_TCHOL'] 


chosen_cols = ['DMDHHSIZ','SEQN', 'RIDEXAGM', 'RIDAGEYR', 'DMDCITZN', 
'SUMTEETH', 'BPXSY1', 'BPXDI1', 'BMXBMI', 'BMXWAIST', 'BMXHT', 
'RIDEXMON', 'RIAGENDR', 'RIDRETH1', 'RIDRETH3', 'DMDEDUC2', 
'DMDBORN4', 'DMDMARTL', 'SIALANG', "DUQ240", "DUQ250", 'PAQ605', 
'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665', 'PAD680', 'SMQ020', 
'SMQ040', 'SMQ900', 'SMD470'] #, 'PHDSESN', 'PHAFSTHRMN', 'PHACOFHRMN'] 
#including the above grey collumns would actually be leakage as they explain
#the chol levels. These will be used for pca in future model

# For future to perform pca on target
# nhanes_df['PHAFSTHRMN'] = nhanes_df['PHAFSTHR'] * 60
# nhanes_df['PHAFSTHRMN'] +=nhanes_df['PHAFSTMN']
# nhanes_df['PHACOFHRMN'] = nhanes_df['PHACOFHR'] * 60
# nhanes_df['PHACOFHRMN'] +=nhanes_df['PHACOFMN']

def make_teeth_column(nhanes_df):
    """Engineers new features regarding teeth

    Args:
        nhanes_df ([pandas DataFrame]): df containing nhanes information

    Returns:
        [DataFrame]: nhanes data with Teeth Feature added.
    """    
    nhanes_df['SUMTEETH'] = 0

    for i in range(1,10):
        nhanes_df[f'OHX0{i}TC'] = nhanes_df[f'OHX0{i}TC'].replace([2], 0)
        for num in range(3, 6):
            nhanes_df[f'OHX0{i}TC'] = nhanes_df[f'OHX0{i}TC'].replace([num], 1)
        nhanes_df['SUMTEETH'] += nhanes_df[f'OHX0{i}TC']

    for i in range(10, 33):
        nhanes_df[f'OHX{i}TC'] = nhanes_df[f'OHX{i}TC'].replace([2], 0)
        for num in range(3, 6):
            nhanes_df[f'OHX{i}TC'] = nhanes_df[f'OHX{i}TC'].replace([num], 1)
        nhanes_df['SUMTEETH'] += nhanes_df[f'OHX{i}TC']
    nhanes_df['PAD680'] = nhanes_df['PAD680'].replace(9999, -1)
    return nhanes_df

nhanes_df = make_teeth_column(nhanes_df)

model_nhanes_df = nhanes_df[chosen_cols]

#rename cols for graph
def importance_columns_renamer(model_nhanes_df):
    """Renames columns for creation of feature importance graph

    Args:
        model_nhanes_df ([DataFrame]): the dataframe used to create
        our random forest model.

    Returns:
        [Dataframe]: [A Dataframe to be used only for graphing puropses.]
    """    
    model_nhanes_df = model_nhanes_df.rename(columns={
                'BMXBMI': 'BMI', 'BMXWAIST ': 'Waist size', "RIDAGEYR": 'Age',
                'BMXHT': 'Height', 'BPXSY1': 'S. Blood Pressure', 
                'BPXDI1': 'D. Blood Pressure', "SUMTEETH": "Missing/Damaged Teeth",
                "DMDHHSIZ": "Household size", "RIDEXMON": 'Season of Data collected', 
                "DMDCITZN": "Citizenship status"
            })
    return model_nhanes_df

def separate_genders(nhanes_df):
    """Splits dataframe by gender

    Returns:
        2 DataFrames: Gendered DataFrames
    """    
    male_df = nhanes_df[nhanes_df['RIAGENDR'] == 1.0]
    female_df = nhanes_df[nhanes_df['RIAGENDR'] == 2.0]
    return male_df, female_df

# target_df.to_csv('/home/riley/Desktop/DSI/capstone2/'+  'chol_crp_target_df.csv')
# model_nhanes_df.to_csv('/home/riley/Desktop/DSI/capstone2/'+ 'demo_quest_exam_feat_df.csv')


model_nhanes_df= model_nhanes_df.set_index('SEQN')

target = pd.DataFrame(target_df[['SEQN', 'HDL_OVER_TCHOL']])
target = target.set_index('SEQN')
target = target.dropna(how='all')

nhanes_features = target.join(model_nhanes_df, on="SEQN", how='left').drop('HDL_OVER_TCHOL', axis=1)
nhanes_features = nhanes_features[nhanes_features['RIDEXAGM'].isnull()]
nhanes_features.drop('RIDEXAGM', axis=1, inplace=True)

target = target.merge(nhanes_features, on='SEQN', how='right')
target = pd.DataFrame(target['HDL_OVER_TCHOL'])

target = target.to_numpy()

for i in nhanes_features.columns:
    nhanes_features[i] = nhanes_features[i].replace([np.nan], -999)



# model = LinearRegression()
# model.fit(cholX_train, choly_train)
# r_sq = model.score(cholX_train, choly_train)
# print('coefficient of determination:', r_sq, "\n")
# r_sq = model.score(cholX_test, choly_test)
# print('coefficient of determination:', r_sq)

# y_pred = model.predict(cholX_test)
# print(model.score(cholX_test, choly_test))
# print(y_pred.min())
# print('predicted response:', y_pred, sep='\n')
def run_random_forest(nhanes_features, target):
    """Runs a random forest model

    Args:
        nhanes_features ([DataFrame]): DataFrame containing features\
            that will be used to predict the target
        target ([Numpy Array]): An array containing the target variable.

    Returns:
        [type]: [description]
    """    
    X_train, X_test, y_train, y_test = train_test_split(\
        nhanes_features, target, random_state=12)
    print('\n random forest')
    target_forest = RandomForestRegressor(n_estimators=600, \
        random_state=8, n_jobs=-1)
    target_forest.fit(X_train, y_train.ravel())
    print(target_forest.score(X_test, y_test.ravel()))
    target_forest.predict(X_test)
    return target_forest

def correct_target_bins(nhanes_df):
    """Creates fake values that cause the bins of the
    cholesterol graph to line up with 0 and 10
    Returns:
        (Pandas DataFrame): temporary version of datframe 
        for graphing purposes only
    """    
    nhanes_df = nhanes_df[nhanes_df['HDL_OVER_TCHOL'] <= 10]
    print(nhanes_df['HDL_OVER_TCHOL'].max())
    nhanes_df.loc[:1 , 'HDL_OVER_TCHOL'] = 0.0
    nhanes_df.loc[1:2 , 'HDL_OVER_TCHOL'] = 0.6
    nhanes_df.loc[2:3 , 'HDL_OVER_TCHOL'] = 10.0
    return nhanes_df


class plot:
    
    def __init__(self, variable, df=nhanes_df, terms=None):
        self.variable = variable
        self.terms = terms
        self.df = df
 

    def plot_importances(self):
        """ Plots importance ranking of top features in Random Forest model
        """        
        n=10
        importances = target_forest.feature_importances_[:n]
        std = np.std([tree.feature_importances_ for tree in target_forest.estimators_],
                    axis=0)
        indices = np.argsort(importances)[::-1]
        features = list(nhanes_features.columns[indices])

        # Print the feature ranking
        print("\n13. Feature ranking:")

        for f in range(n):
            print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))

        # Plot the feature importances of the forest
        fig, ax = plt.subplots(figsize=(10,15))

        ax.bar(range(n), importances[indices], yerr=std[indices], color="palevioletred", align="center")
        ax.set_xticks(range(n))
        ax.set_xticklabels(features, rotation = 90)
        ax.set_xlim([-1, n])
        ax.set_xlabel("Importance")
        plt.xticks(rotation=30, ha='right')
        ax.set_title("Feature Importances")
        plt.show()


    def plot_variable_hist(self, colors='Set3', bin_size=32):
        """Takes in specifications and produces a graph

        Args:
            colors (str, optional): colors to be assigned to "pallete" feature in \
                seaborn. Defaults to 'Set3'.
            bin_size (int, optional): Histogram bin size. Defaults to 32.
        """        
        ax = sns.histplot(x=self.variable, data=self.df, palette=colors, bins=bin_size)
        ax.set_xlabel(self.terms['xlabel'])
        ax.set_ylabel(self.terms['ylabel'])
        if self.variable =='RIDRETH3':
            ax.set_xlim([0,10]) 
            ax.set_xticks([0, .5, 1.0, 1.5, 2.0, 2.5, 3.0, 
            3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
        ax.set_title(self.terms['title'])
        plt.show()

    def sbcountplot(self, colors="Set3" ):
        """Creats a histogram-like barchart that sums categories in a given column

        Args:
            colors (str, optional): [description]. Defaults to "Set3".
        """        
        ax = sns.countplot(x=self.variable, data=self.df, palette=colors, order = nhanes_df[self.variable].value_counts().index)
        ax.set_xlabel(self.terms['xlabel'])
        ax.set_ylabel(self.terms['ylabel'])
        ax.set_title(self.terms['title'])
        if self.variable =='RIDRETH3':
            plt.xticks(rotation=30, ha='right') # for ethincity plot
        plt.show()  



if __name__ == "__main__":
    # pass
    run_random_forest(nhanes_features, target)

    feature_importances = plot('importances', dict())
    # feature_importances.plot_importances()

    age_dic = {
        'title': 'Age Of Study Participants',
        'xlabel': 'Years in 5 year bins',
        'ylabel': 'Number of People'
    }
    age_plot = plot('RIDAGEYR', terms=age_dic)
    # age lim was 80 so bin size of16 creates five year periods
    # age_plot.plot_variable_hist(bin_size=16)

    #remove children
    model_nhanes_df = model_nhanes_df[model_nhanes_df['RIDEXAGM'].isnull()]
    # nhanes_df = nhanes_df[nhanes_df['RIDAGEYR'].notnull()]

    # plot missingno
    # msno.matrix(model_nhanes_df)
    # plt.show()

    nhanes_df['Gender'] = nhanes_df['RIAGENDR']
    nhanes_df['Gender'].replace([1.0], 'Male', inplace=True)
    nhanes_df['Gender'].replace([2.0], 'Female', inplace=True)

    gender_dic = {
        'title': 'Gender of Adult Study Participants',
        'xlabel': '',
        'ylabel': 'Number of people'
    }
    gender_plot = plot('Gender', df=nhanes_df, terms=gender_dic)
    # gender_plot.sbcountplot(colors=['pink','skyblue'])

    nhanes_df['RIDRETH3'].replace([3.0], 'Caucasian', inplace=True)
    nhanes_df['RIDRETH3'].replace([4.0], 'Black', inplace=True)
    nhanes_df['RIDRETH3'].replace([1.0], 'Mexican American', inplace=True)
    nhanes_df['RIDRETH3'].replace([2.0], 'Hispanic Other', inplace=True)
    nhanes_df['RIDRETH3'].replace([6.0], 'Asian', inplace=True)
    nhanes_df['RIDRETH3'].replace([7.0], 'Other Race or Multi-Racial',
    inplace=True)

    ethnicity_dic = {
        'title': 'Ethnicity of Adult Study Participants',
        'ylabel': 'Number of People',
        'xlabel': 'Ethnicities'
    }
    ethnicity_plot = plot('RIDRETH3', df=nhanes_df, terms=ethnicity_dic)
    ethnicity_plot.sbcountplot()


    tooth_dic = {
        'title': 'Histogram of Adults Missing Teeth',
        'xlabel': 'Total Teeth Missing/Severely Damaged Per Person',
        'ylabel': 'Number of People'
    }
    tooth_plot = plot('SUMTEETH', terms=tooth_dic)
    # tooth_plot.plot_variable_hist()

    target_dic = {
        'title': 'Histogram of Cholesterol values',
        'ylabel': 'People per Bin',
        'xlabel': 'Ratio of Total Cholesterol to HDL'
    }
    target_plot = plot('HDL_OVER_TCHOL', terms=target_dic)
    # target_plot.plot_variable_hist(bin_size=20)



