import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib as mpl
from feat_engineer import 
mpl.rcParams['figure.dpi'] = 200

class datapipeline:

    def __init__(self):  # , df, df_dic, target, features):
        self.df = pd.DataFrame()
        self.df_dic = dict()
        self.target = pd.DataFrame()
        self.features = pd.DataFrame()

    def load_NHANES_data(self):
        """Loads nhanes data by splitting a strings containing lists of files
        indto individual file names and then converting those files
        from XPT format into a pandas DataFrame then merging appropriate
        files into output Dataframe

        Extra files are left in code for ease of addition into main DataFrame

        Returns:
            [Dict]: A dictionary containing a DataFrame for each file used
            from the 2017-2018 data.
        """
        lab = "HSCRP_J.XPT HDL_J.XPT TCHOL_J.XPT FASTQX_J.XPT"
        more_lab = "ALB_CR_J.XPT GLU_J.XPT CRCO_J.XPT FOLFMS_J.XPT \
        HEPB_S_J.XPT IHGEM_J.XPT UCM_J.XPT VIC_J.XPT BIOPRO_J.XPT \
        GHB_J.XPT HEPC_J.XPT INS_J.XPT UCPREG_J.XPT VOCWB_J.XPT CBC_J.XPT \
        FERTIN_J.XPT HEPE_J.XPT PBCD_J.XPT UHG_J.XPT CMV_J.XPT \
        FETIB_J.XPT HEPA_J.XPT HIV_J.XPT UIO_J.XPT COT_J.XPT FOLATE_J.XPT \
        HEPBD_J.XPT HSCRP_J.XPT UCFLOW_J.XPT UNI_J.XPT"
        questionnaire = "DUQ_J.XPT PAQ_J.XPT SMQ_J.XPT SMQFAM_J.XPT \
        DBQ_J.XPT MCQ_J.XPT WHQ_J.XPT"
        # ADDITIONASL QUESTIONNAIRE: ACQ_J.XPT  DEQ_J.XPT  ECQ_J.XPT  IMQ_J.XPT
        # OSQ_J.XPT RHQ_J.XPT  SMQFAM_J.XPT  WHQMEC_J.XPT AUQ_J.XPT  DIQ_J.XPT
        # HEQ_J.XPT  KIQ_U_J.XPT  PAQ_J.XPT  RXQASA_J.XPT  SMQ_J.XPT BPQ_J.XPT
        # DLQ_J.XPT  HIQ_J.XPT  MCQ_J.XPT  PAQY_J.XPT
        # RXQ_DRUG.xpt  SMQRTU_J.XPT CDQ_J.XPT  DPQ_J.XPT
        # HSQ_J.XPT  OCQ_J.XPT   PFQ_J.XPT   RXQ_RX_J.XPT
        # SMQSHS_J.XPT  DUQ_J.XPT  HUQ_J.XPT  OHQ_J.XPT
        # PUQMEC_J.XPT  SLQ_J.XPT  WHQ_J.XPT"
        demo = "DEMO_J.XPT"
        diet = "DR1TOT_J.XPT DR2TOT_J.XPT DS1TOT_J.XPT \
        DS2TOT_J.XPT DSQTOT_J.XPT"
        exam = 'BMX_J.XPT BPX_J.XPT OHXDEN_J.XPT'
        # THE FOLLOWING WERE MORE EXPENSIVE THAN A BLOOD TEST
        # OHXREF_J.XPT DXXFEM_J.XPT DXX_J.XPT DXXSPN_J.XPT LUX_J.XPT'

        lst = [demo, exam, questionnaire, diet, lab, more_lab]

        path = "/home/riley/Desktop/DSI/capstone2/"
        file_locations = ['demographics/', 'exam/', 'questionnaire_/',
                          'dietary/', 'lab/', 'lab/morelab/']

        for idx, i in enumerate(lst):
            i = i.replace("    ", " ")
            i = i.replace("   ", " ")
            i = i.replace("  ", " ")
            i = i.split(" ")
            lst[idx] = i

        for idx, folder in enumerate(lst):
            for file in folder:
                file_path = f'{path}{file_locations[idx]}{file}'
                self.df_dic[file] = pd.read_sas(file_path, format='xport',
                                                encoding='utf-8')
        return lst

    def merge_dataframes(self):
        lst = self.load_NHANES_data()
        lab_df = self.df_dic['HDL_J.XPT']
        lab_df = lab_df.merge(self.df_dic['TCHOL_J.XPT'], on='SEQN',
                              how='outer')
        lab_df = lab_df.merge(self.df_dic['FASTQX_J.XPT'], on='SEQN',
                              how='outer')

        x_data = lst[:4]

        self.df = lab_df.copy()
        for folder in x_data:
            for file in folder:
                self.df = self.df.merge(self.df_dic[file], on='SEQN',
                                        how='left')

        # self.df = self.df.merge(self.df_dic['DUQ_J.XPT'], on='SEQN',
        #                         how='left')
        # self.df = self.df.merge(self.df_dic['PAQ_J.XPT'], on='SEQN',
        #                         how='left')
        # self.df = self.df.merge(self.df_dic['SMQ_J.XPT'], on='SEQN',
        #                         how='left')
        # self.df = self.df.merge(self.df_dic['SMQFAM_J.XPT'],
        #                         on='SEQN', how='left')
        self.df = self.df[self.df['RIDEXAGM'].isnull()]
        self.df = self.df[self.df['LBXTC'].notna()]
        self.df = self.df[self.df['LBDHDD'].notna()]
        # self.df = self.df[self.df['LBDHDD'] < 135]
        # self.df = self.df[self.df['LBDHDD'] > 17]
        # self.df = self.df[self.df['LBXTC'] < 340]
        return self.df

    def remove_children(self):
        self.df = self.df[self.df['RIDEXAGM'].isnull()]
        return self.df

    def make_teeth_column(self):
        """Engineers new features regarding teeth

        Args:
            self: predominately uses class Dataframe to

        Returns:
            [DataFrame]: nhanes data with Teeth Feature added.
        """
        self.df['SUMTEETH'] = 0

        for i in range(1, 10):
            self.df[f'OHX0{i}TC'] = self.df[f'OHX0{i}TC'].replace([2], 0)
            for num in range(3, 6):
                self.df[f'OHX0{i}TC'] = self.df[f'OHX0{i}TC'].replace([num], 1)
            self.df['SUMTEETH'] += self.df[f'OHX0{i}TC']

        for i in range(10, 33):
            self.df[f'OHX{i}TC'] = self.df[f'OHX{i}TC'].replace([2], 0)
            for num in range(3, 6):
                self.df[f'OHX{i}TC'] = self.df[f'OHX{i}TC'].replace([num], 1)
            self.df['SUMTEETH'] += self.df[f'OHX{i}TC']
        # self.df['PAD680'] = self.df['PAD680'].replace(9999, -1)
        return self.df

    def create_fitness_score(self):
        """[Creates feature that calculates minutes per week performing
        each category of activity as well as adjusting for physiological
        impact. ex: Vigorous activity has much higher impact than other
        types of exercise and so is weighted strongly]

        Returns:
            [df]: [Original df with feature added]
        """        
        self.df['FITNESS_SCORE_DAY'] = 0
        self.df.loc[self.df['PAQ610'].isin(range(1, 8)),
                    "FITNESS_SCORE_DAY"] += self.df.loc[
                    self.df['PAQ610'].isin(range(1, 8)), 'PAQ610'] * 5

        self.df.loc[self.df['PAQ655'].isin(range(1, 8)),
                    "FITNESS_SCORE_DAY"] += self.df.loc[
                    self.df['PAQ655'].isin(range(1, 8)), 'PAQ655'] * 5

        self.df.loc[self.df['PAQ625'].isin(range(1, 8)),
                    "FITNESS_SCORE_DAY"] += self.df.loc[
                    self.df['PAQ625'].isin(range(1, 8)), 'PAQ625'] * 2

        self.df.loc[self.df['PAQ670'].isin(range(1, 8)),
                    "FITNESS_SCORE_DAY"] += self.df.loc[
                    self.df['PAQ670'].isin(range(1, 8)), 'PAQ670'] * 2

        self.df.loc[self.df['PAQ640'].isin(range(1, 8)),
                    "FITNESS_SCORE_DAY"] += self.df.loc[
                    self.df['PAQ640'].isin(range(1, 8)), 'PAQ640']

        self.df['FITNESS_SCORE_MIN'] = 0
        self.df.loc[self.df['PAD615'].isin(range(10, 840)),
                    "FITNESS_SCORE_MIN"] += self.df.loc[
                    self.df['PAD615'].isin(range(10, 840)),
                    'PAQ610'] * 6 * self.df.loc[self.df[
                     'PAD615'].isin(range(10, 840)), 'PAD615']

        self.df.loc[self.df['PAD660'].isin(range(10, 840)),
                    "FITNESS_SCORE_MIN"] += self.df.loc[
                    self.df['PAD660'].isin(range(10, 840)),
                    'PAQ655'] * 6 * self.df.loc[self.df[
                     'PAD660'].isin(range(10, 840)), 'PAD660']

        self.df.loc[self.df['PAD630'].isin(range(10, 840)),
                    "FITNESS_SCORE_MIN"] += self.df.loc[
                    self.df['PAD630'].isin(range(10, 840)),
                    'PAQ625'] * 2 * self.df.loc[self.df[
                     'PAD630'].isin(range(10, 840)), 'PAD630']

        self.df.loc[self.df['PAD675'].isin(range(10, 840)),
                    "FITNESS_SCORE_MIN"] += self.df.loc[
                    self.df['PAD675'].isin(range(10, 840)),
                    'PAQ670'] * 2 * self.df.loc[self.df[
                     'PAD675'].isin(range(10, 840)), 'PAD675']

        self.df.loc[self.df['PAD645'].isin(range(10, 840)),
                    "FITNESS_SCORE_MIN"] += self.df.loc[
                    self.df['PAD645'].isin(range(10, 840)),
                    'PAQ640'] * 1 * self.df.loc[self.df[
                     'PAD645'].isin(range(10, 840)), 'PAD645']

        self.df.loc[self.df['PAD680'].isin(range(10, 1320)),
                    "FITNESS_SCORE_MIN"] -= self.df.loc[
                    self.df['PAD680'].isin(range(10, 1320)),
                    'PAD680'] * 1

        return self.df

    def blood_pressure(self):
        """[Creates blood pressure feature]
        """        
        self.df['BP_DIFF'] = self.df['BPXSY1'] - self.df['BPXDI1']

    def fasting_features(self):
        """[Creates feature that shows how many minutes a patient fasted]
        """        
        self.df['PHAFSTHRMN'] = self.df['PHAFSTHR'] * 60
        self.df['PHAFSTHRMN'] += self.df['PHAFSTMN']
        self.df['PHACOFHRMN'] = self.df['PHACOFHR'] * 60
        self.df['PHACOFHRMN'] += self.df['PHACOFMN']

    def impute_values(self):
        """[Allows for imputation of values in numeric columns]
        """        
        numeric_cols = ['SUMTEETH', 'BPXSY1', 'BPXDI1', 'BMXBMI', 'BMXWAIST',
                        'BMXHT', 'RIDAGEYR', 'BMXARML', 'BMXARMC', 'BMXLEG']

        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer = imputer.fit(self.df[numeric_cols])
        imputed_df = imputer.transform(self.df[numeric_cols].values)
        for idx in range(len(numeric_cols)):
            self.df[numeric_cols[idx]] = imputed_df[:, idx]
        return self.df

    def replace_nans(self):
        """[alternative strategy to imputing values for dealing with NaN]
        """        
        for i in self.df.columns:
            self.df[i] = self.df[i].replace([np.nan], -2)

    def create_target(self):
        """[Creates Cholesterol related target values. Both forms of the 
        Total Cholesterol/HDL ratio are available]
        """
        self.df['LBXTC'] = self.df['LBXTC'].replace(-2, 1)
        self.target['HDL_OVER_TCHOL'] = self.df['LBDHDD']/self.df['LBXTC']
        self.target['TCHOL_OVER_HDL'] = self.df['LBXTC']/self.df['LBDHDD']
        self.target['SEQN'] = self.df['SEQN']
        self.target = self.target.set_index('SEQN')
        self.target = self.target  # .to_numpy()
        return self.target

    def create_alt_target(self):
        """[Creation of alternative target that predicts heart disease]
        WARNING model not yet optimized for use.
        """
        self.target['MCQ160C'] = self.df['MCQ160C']
        self.target['SEQN'] = self.df['SEQN']
        self.target = self.target.set_index('SEQN')
        return self.target

    def create_features(self):
        """[Allows for creation of features as well as ability to switch
        between feature choices, used in random forest model. Switching is
        used to compare and plot.]
        """        
        self.make_teeth_column()
        self.create_fitness_score()
        self.fasting_features()
        self.blood_pressure()
        abrv_cols = ['SEQN', 'BMXWAIST', 'BMXBMI', 'BMXARMC', 'RIAGENDR',
                     'WHD140', 'BPXDI1', 'WHQ150', 'BMXHT', 'BMXLEG',
                     'DR1TCHOL', 'BPXSY1', 'DR1TPFAT', 'BMXARML',
                     'FITNESS_SCORE_MIN', 'SUMTEETH', 'RIDAGEYR', 'PAD680']
        chosen_cols1 = ['DMDHHSIZ', 'SEQN', 'RIDAGEYR', 'DMDCITZN',
                        'SUMTEETH', 'BPXSY1', 'BPXDI1', 'BMXBMI', 'BMXWAIST',
                        'RIDEXMON', 'RIAGENDR', 'RIDRETH3',
                        'DMDBORN4', 'DMDMARTL', 'SIALANG', "DUQ240", "DUQ250",
                        'PAQ605', 'PAQ620', 'PAQ635', 'PAQ650', 'PAQ665',
                        'PAD680', 'SMQ020', 'SMQ040', 'SMQ900', 'SMD470',
                        'BMXARML', 'BMXARMC', 'BMXLEG', 'BMXHT',  'DMDEDUC2',
                        'PHAFSTHRMN', 'DR1TTFAT', 'DR1TCHOL', 'DS1DSCNT',
                        'DR1TMFAT', 'DR1TPFAT', 'DR1TSODI', 'DR1TALCO',
                        'DS1TVD', 'WHD140', 'WHQ150']  # , 'FITNESS_SCORE_MIN']
        abrv_cols2 = ['SEQN', 'BMXWAIST', 'BMXBMI', 'BMXARMC', 'RIAGENDR',
                      'WHD140', 'BPXDI1', 'WHQ150', 'BMXHT', 'BMXLEG',
                      'DR1TCHOL', 'BPXSY1', 'DR1TPFAT', 'BMXARML',
                      'SUMTEETH', 'RIDAGEYR', 'PAQ605', 'PAQ620', 'PAQ635',
                      'PAD680', 'PAQ665', 'PAQ650'] 
        chosen_cols = abrv_cols

        # The following were initially not useful. may require adj. for model.
        # They also may not be useful at all further investigation needed.
        #    , 'DR1DRSTZ_x', 'DR1DRSTZ_y', 'DR2DRSTZ_x',
        #    'DR2DRSTZ_y', 'OHDDESTS', 'OHDEXSTS', 'BMIHIP',
        #    'DR2TPROT', 'DR2TPHOS', 'DR2TZINC', 'DR2TATOC',
        #    'DR2TS160', 'DR2TMAGN', 'DR2TIRON', 'DR2TKCAL']

        self.features = self.df[chosen_cols]
        # print(self.df.info())
        self.features = self.features.set_index('SEQN')
        return self.features, self.df

    def one_hot(self):
        """one hot encodes the relevant features.
        """
        variables = ['RIDRETH3', 'DMDMARTL', 'DMDEDUC2']
        X = self.features
        one_hot_df = pd.get_dummies(X, columns=variables, drop_first=True)
        return one_hot_df

    def create_rf_X_y(self, model=True):
        
        clean_nhanes = datapipeline()
        clean_nhanes.merge_dataframes()
        # clean_nhanes.remove_children()
        clean_nhanes.replace_nans()
        target = clean_nhanes.create_target()  # add alt for changed target
        features, df = clean_nhanes.create_features()
        return target, features, df


if __name__ == "__main__":
    clean_nhanes = datapipeline()
    target, features, df = clean_nhanes.create_rf_X_y(model=False)
