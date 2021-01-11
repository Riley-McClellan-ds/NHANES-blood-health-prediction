import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import (cluster, datasets, decomposition, ensemble, manifold,
                     random_projection, preprocessing)
import missingno as msno
from plot import plot
from sklearn.impute import SimpleImputer
from math import e
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score)
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import RandomizedSearchCV
from matplotlib import cm
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from random import randint
import xgboost as xgb
from xgboost import plot_tree
import statistics as sts
from clean_data import datapipeline
from random import seed
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def import_rf_nhanes(testing=False):
    """[Uses datapipeline from the clean_data.py file to import and clean
    the 2017-2018 NHANES data, handling feature engineering and isolating
    target and features.]

    Args:
        testing (bool, optional): [Adds "Random Numbers" column to target].
        Defaults to False.

    Returns:
        Target, Features, df: Pandas Dataframes containing 
        Target, chosen Features as well as the complete NHANES dataframe
    """    
    clean_nhanes = datapipeline()
    target, features, df = clean_nhanes.create_rf_X_y()
    # target = target['HDL_OVER_TCHOL']  # .apply(lambda x: np.log(x+1))
    if testing:
        features['random_numbers'] = np.random.randint(1, 1000000,
                                                       size=features.shape[0])
    return target, features, df


def make_categorical_multi(target):
    """[Creates classes for classification models with multi-output]

    Args:
        target ([Pandas Dataframe]): [Dataframe used to hold target info]

    Returns:
        target: [target with classes added]
    """    
    target['CHOL_RISK'] = 0
    target.loc[target['TCHOL_OVER_HDL'] > 5.0, 'CHOL_RISK'] = 2

    # target.loc[target['TCHOL_OVER_HDL'] > 3.5, 'CHOL_RISK'] = 1  # 3.74
    # target.loc[target['TCHOL_OVER_HDL'] > 4.85, 'CHOL_RISK'] = 2  # 4.85
    # target.loc[target['TCHOL_OVER_HDL'] > 7.5, 'CHOL_RISK'] = 3 # 5.67
    target.loc[target['TCHOL_OVER_HDL'] > 8.0, 'CHOL_RISK'] = 4  # 8.42
    return target


def make_categorical_binary(target):
    """[Creates classes for classification models with binary output]

    Args:
        target ([Pandas Dataframe]): [Dataframe used to hold target info]

    Returns:
        target: [target with classes added]
    """    
    target['CHOL_RISK'] = 0
    # target.loc[target['TCHOL_OVER_HDL'] > 3.5, 'CHOL_RISK'] = 1
    target.loc[target['TCHOL_OVER_HDL'] > 5.0, 'CHOL_RISK'] = 1

    # print(len(target[target['CHOL_RISK'] == 4]))
    return target


def run_rf_gridsearch_c(features, target, oversample=False, ts=.25):
    """[Searches random forest for optimal parameters]

    Args:
        features ([Pandas DataFrame]): [Random forest features]
        target ([Pandas DataFrame]): [Random forest target]
        oversample (bool, optional): [Allows oversampling]. Defaults to False.
        ts (float, optional): [Test Size]. Defaults to .25.
    """
    X_train, _, y_train, _ = train_test_split(
        features, target, stratify=target, test_size=ts, random_state=10)
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    grid = {'max_depth': [None],
            'max_features': ['sqrt', 'log2', None],
            'min_samples_split': [15, 16, 17],
            'min_samples_leaf': [9, 10, 11],
            'bootstrap': [True, False],
            'n_estimators': [8000, 9000],
            'random_state': [1, 12, 15, 32, 69]}

    gridsearch = RandomizedSearchCV(RandomForestClassifier(),
                                    grid, scoring='f1', n_iter=20,
                                    n_jobs=-1, cv=5, verbose=4)
    gridsearch.fit(X_train, y_train)
    # The best model itself, which you can run .predict/.predict_proba
    model = gridsearch.best_estimator_
    # the gridsearch object has some further info
    print(f'Best Params: {gridsearch.best_params_}')
    print(f'Best F1 Score: {gridsearch.best_score_:.3f}')
    return model


def run_rfr(features, target):
    """Runs a random forest model

    Args:
        features ([DataFrame]): DataFrame containing features
        that will be used to predict the target
        target ([Numpy Array]): An array containing the target variable.

    Returns:
        [target_forest]: [Random forest model fit to input features and target]
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, random_state=12)
    target_forest = RandomForestRegressor(n_estimators=600, criterion='mse',
                                          random_state=8, n_jobs=-1)
    target_forest.fit(X_train, y_train.to_numpy().ravel())
    preds = target_forest.predict(X_test)
    mse = cross_val_score(target_forest, X_train, y_train.to_numpy().ravel(),
                          scoring='neg_root_mean_squared_error',
                          cv=5, n_jobs=-1) * -1
    print('Random forest Regressor STATS:\n')
    print('mse:', mse)
    return target_forest


def run_rfc_multioutput(features, target, oversample=False, ts=.25):
    """[summary]

    Args:
        features ([type]): [description]
        target ([type]): [description]
        oversample (bool, optional): [description]. Defaults to False.
        ts (float, optional): [description]. Defaults to .25.

    Returns:
        [type]: [description]
    """    
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, stratify=target, test_size=ts, random_state=10)
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    rfc = RandomForestClassifier(random_state=32, n_estimators=2500,
                                 min_samples_split=15, min_samples_leaf=69)
    rfc.fit(X_train, y_train)
    y_predict = rfc.predict(X_test)

    # !The following allows adjustment of probability for 3 classes only
    # y_predict_proba = rfc.predict_proba(X_test)
    # y_predict = switch_proba(y_predict_proba, threshold=.50001, floor=.5)

    print('RF CLASSIFIER MULTI-OUTPUT STATS:')
    print("\n confusion matrix:\n", confusion_matrix(y_test, y_predict))
    print("\n f1 score ", f1_score(y_test, y_predict, average='weighted'))
    return rfc


def switch_proba(arr, threshold=.98, floor=.5):
    """[Allows for adjustment of probability for 3 target output only]

    Args:
        arr ([array]): [Array of probabilities for each class 3 columns]
        threshold (float, optional): [Probabilities lesser than this value in
            class "0" are swapped with the greater of the remaining
            two probabilites if also greater floor value]. Defaults to .98.
        floor (float, optional): [Probabilities above this value in the
            "0" class are swapped with the greater of the remaining
            two probabilites if also less than threshold
            value]. Defaults to .5.

    Returns:
        [type]: [description]
    """
    new_arr = arr.copy()
    temp_arr = arr.copy()
    # print(new_arr[:10, :], '\n')
    col0 = temp_arr[:, 0]
    col1 = temp_arr[:, 1]
    col2 = temp_arr[:, 2]
    col0b = (col0 < threshold) & (col0 > floor)
    col1b = col1 < col2

    new_arr[col0b, col1b[col0b]+1] = temp_arr[col0b, 0]
    new_arr[col0b, 0] = temp_arr[col0b, col1b[col0b]+1]
    lst = []
    for i in new_arr:
        lst.append(np.argmax(i))
    return np.array(lst)


def run_rfc_binaryoutput(features, target, oversample=False, ts=.25):
    """[Creates binary output random forest model using NHANES data]

    Args:
        features ([pandas DataFrame]): [Features for model]
        target ([pandas Series]): [pandas Series containing cholesterol in
                the form of two classes]
        oversample (bool, optional): [Oversamples data when True]. 
                                      Defaults to False.
        ts (float, optional): [Test size]. Defaults to .25.

    Returns:
        [rfc]: [Binary random forest model]
    """    
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, stratify=target, test_size=ts, random_state=10)
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)
    rfc = RandomForestClassifier(random_state=32, n_estimators=2500,
                                 min_samples_split=15, min_samples_leaf=69,
                                 class_weight='balanced_subsample')
    rfc.fit(X_train, y_train)
    # print("\n score:", rfc.score(X_test, y_test))
    # y_predict = rfc.predict(X_test)

    y_predict_proba = rfc.predict_proba(X_test)[:, 1]
    y_predict = y_predict_proba > .475

    f1 = cross_val_score(rfc, X_test, y_test,
                         scoring='f1',
                         cv=6, n_jobs=-1)
    print('\nRF CLASSIFIER BINARY OUTPUT STATS:')
    print("\n cross val f1", sts.mean(f1))
    print("\n confusion matrix:\n", confusion_matrix(y_test, y_predict))
    print("\n f1 score ", f1_score(y_test, y_predict))
    return rfc


def run_xgb_regressor(features, target, ts=.25):
    """[Creates an XGBoost regression tree and prints RMSE score]

    Args:
        features ([pandas DataFrame]): [Features for model]
        target ([pandas Series]): [pandas Series containing cholesterol in
                 its original ratio form.]
        ts (float, optional): [Test size]. Defaults to .25.

    Returns:
        [xgb_r]: [XGBoost regression model fit to input data]
    """    
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=ts, random_state=10)
    xgb_r = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=100, seed=69, eval_metric='mae')

    xgb_r.fit(X_train, y_train)

    pred = xgb_r.predict(X_test)
    z = y_test.to_numpy()
    x = np.argwhere(z > 5.0)
    y = []
    for i in x:
        y.append(i[0])
    for i in y:
        if pred[i] > 5.0:
            print(list(pred)[i])
    # below code showed only 25 values correctly predicted over 5
    # print(len(pred[pred > 5.0]))
    # for i in pred[:10]:
    #     print(i)
    rmse = np.sqrt(mse(y_test, pred))
    print('XGB Regressor STATS:\n')
    print("RMSE : % f" % (rmse))
    return xgb_r


def run_xgb_classifier(features, target, oversample=False, ts=.25):
    """[Binary classifer using XGBoost tree]

    Args:
        features ([pandas DataFrame]): [Features for model]
        target ([pandas Series]): [pandas Series containing cholesterol in
                the form of two classes]
        oversample (bool, optional): [Oversamples data when True]. 
                                      Defaults to False.
        ts (float, optional): [Test size]. Defaults to .25.

    Returns:
        [xgb_c]: [Binary XGBoost classification model]
    """    
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, stratify=target, test_size=ts, random_state=10)
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    xgb_c = xgb.XGBClassifier(n_estimators=500, seed=69)
    xgb_c.fit(X_train, y_train)
    y_predict = xgb_c.predict(X_test)

    y_predict_proba = xgb_c.predict_proba(X_test)[:, 1]
    y_predict = y_predict_proba > .03

    f1 = cross_val_score(xgb_c, X_test, y_test,
                         scoring='f1',
                         cv=6, n_jobs=-1)
    print('XGB CLASSIFIER STATS:\n')
    print("\n cross val f1", sts.mean(f1))
    print("\n confusion matrix:\n", confusion_matrix(y_test, y_predict))
    print("\n f1 score ", f1_score(y_test, y_predict))  # ,average='weighted'))

    # multi:softmax: set XGBoost to do multiclass
    # classification using the softmax objective,
    # you also need to set num_class(number of classes)
    return xgb_c


def run_xgb_c_multioutput(features, target, oversample=False, ts=.25):

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, stratify=target, test_size=ts, random_state=10)
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

    xgb_c = xgb.XGBClassifier(n_estimators=1000, random_state=69,
                              objective='multi:softmax', num_class=3)
    xgb_c.fit(X_train, y_train)
    # y_predict = xgb_c.predict(X_test)

    y_predict_proba = xgb_c.predict_proba(X_test)  # [:, 1]
    y_predicts = switch_proba(y_predict_proba, threshold=.5, floor=.499999)
    # print(type(y_predict))
    # print(y_predicts[:10, :])
    lst = []
    for i in y_predicts:
        lst.append(np.argmax(i))
    y_predict = np.array(lst)
    print('XGB CLASSIFIER MULTI-OUTPUT STATS:\n')
    print("\n confusion matrix:\n", confusion_matrix(y_test, y_predict))
    print("\n f1 score ", f1_score(y_test, y_predict, average='weighted'))
    return xgb_c


def scree_plot(ax, pca, n_components_to_plot=30, title=None):
    """[Make a scree plot showing the variance explained (i.e. 
    variance of the projections) for the principal components 
    in a fit sklearn PCA object.]
    
    Args:

        ax [matplotlib.axis]: The axis to make the scree plot on.
        
        pca [sklearn.decomposition.PCA]: A fit PCA object.
        
        n_components_to_plot [int]: The number of principal 
                components to display in the skree plot.
        
        title [str]: A title for the skree plot.
    """
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    ax.plot(ind, vals, color='blue')
    ax.scatter(ind, vals, color='blue', s=50)

    for i in range(num_components):
        ax.annotate(r"{:2.2f}".format(vals[i]),
                   (ind[i]+0.2, vals[i]+0.005),
                   va="bottom",
                   ha="center",
                   fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)
    ax.set_ylim(0, max(vals) + 0.05)
    ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=16)
    plt.show()

# Unused code from EDA preserved for future use
def pca_testing():
    """
    Testing that went unused, code preserved for future investigation
    """
    # ss = preprocessing.StandardScaler()
    # X_centered = ss.fit_transform(df)
    # pca = decomposition.PCA(n_components=2)
    # X_pca = pca.fit_transform(X_centered)
    # breakpoint()
    # print(X_pca)
    # lst = []
    # for _ in range(10):
    #     x = np.argmax(pca.components_[2])
    #     lst.append(x)
    #     pca.components_[2][x] = 0c
    # for i in lst:
    #     print(df.columns[i])
    # X_pca1 = X_pca[lambda x: x[: , 0]]
    # features['pca1'] = X_pca[:, 0]
    # features['pca2'] = X_pca[:, 1]
    pass


def plot_pca_embedding(ax, X, y, title=None):
    """Plot an embedding of the NHANES pca onto a plane.
    
    Args:
        ax [matplotlib.axis object]: The axis to make the scree plot on.
        
        X [numpy.array, shape (n, 2)]: A two dimensional array
                containing the coordinates of the embedding.
        
        y [numpy.array]: The labels of the datapoints. Should be digits.

        title [str]: A title for the plot.
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], 
                 str(y[i]), 
                 color=plt.cm.Set1(y[i] / 10.), 
                 fontdict={'weight': 'bold', 'size': 12})

    # ax.set_xticks([]),
    # ax.set_yticks([])
    # ax.set_ylim([-0.1,1.1])
    # ax.set_xlim([-0.1,1.1])

    if title is not None:
        ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    target, features, df = import_rf_nhanes(testing=False)
    features['age_max_weight'] = features['RIDAGEYR'] - features['WHQ150']
    # features['max_weight_by_age'] *= features['WHD140']
    features =  features.drop('WHQ150', axis=1)
    # features =  features.drop('WHD140', axis=1)
    #!
    # bst = run_xgb_regressor(features, target['TCHOL_OVER_HDL'])

    thing = ['BMXWAIST', 'BMXBMI', 'BMXARMC', 'RIAGENDR', 'RIDAGEYR',
             'WHD140', 'BPXDI1', 'BMXHT', 'BMXLEG',
             'DR1TCHOL', 'BPXSY1', 'DR1TPFAT', 'BMXARML',
             'FITNESS_SCORE_MIN', 'SUMTEETH', 'age_max_weight']
    # print(features.loc[94607.0, thing])
    # print(features.loc[99848.0, thing])

    # rf = run_rfr_multioutput(features, target['TCHOL_OVER_HDL'])

    # feat_imp = plot('cheese', forest=rf, forest_feat=features)
    # feat_imp.plot_importances(n=len(features.columns))
    #!

    # # total_df['MCQ160C'] = target['MCQ160C']

    # run_rf_gridsearch(features, total_df['CHOL_RISK'], oversample=False, ts=.30)


    # target = make_categorical_binary(target)
    # xgb_c = run_xgb_classifier(features, target['CHOL_RISK'],
    #                          oversample=True, ts=.2)

    # target = make_categorical_multi(target)
    # xgb_c = run_xgb_c_multioutput(features, target['CHOL_RISK'],
    #                             oversample=True, ts=.3)

    target = make_categorical_multi(target)
    rf = run_rfc_multioutput(features, target['CHOL_RISK'],
                             oversample=True, ts=.20)
    feat_imp = plot('cheese', df=features, forest=rf, forest_feat=features)
    feat_imp.plot_importances(n=len(features.columns))
    # feat_imp.plot_importances(n=10)

    # target = make_categorical_binary(target)
    # rf = run_rfc_binaryoutput(features, target['CHOL_RISK'],
    #                          oversample=True, ts=.2)
    # feat_imp = plot('cheese', forest=rf, forest_feat=features)
    # !feat_imp.plot_importances(n=len(features.columns))

    # feat_imp = plot('cheese', forest=rf, forest_feat=features)
    # feat_imp.plot_importances(n=20)

    # drop_cols = ['LBDHDDSI', 'LBDTCSI', 'LBDHDD', 'LBXTC']
    # for i in drop_cols:
    #     df.drop(i, axis=1, inplace=True)
    # for i in df.columns:
    #     if type(df.loc[0, i]) == str:
    #         df.drop(i, axis=1, inplace=True)
    # df = df.set_index('SEQN')

    # for i in df.columns:
    #         df[i] = df[i].replace([np.nan], -9)

    # c = df.corrwith(target, axis=1).abs()
    # # # print(type(c))
    # # # s = c.unstack()
    # # # so = s.sort_values(kind="quicksort")

    # # print(c.sort_values(ascending=True))

    # rf = run_rfc(features, target['CHOL_RISK'], oversample=True)
    # print(type(rf.feature_importances_[0]))
    # feat_imp = plot('cheese', forest=rf, forest_feat=features)
    # feat_imp.plot_importances(n=len(features.columns))

    # fig, ax = plt.subplots(figsize=(10, 6))
    # scree_plot(ax, pca, title="Scree Plot for Digits Principal Components")

    # pca = decomposition.PCA(n_components=5)
    # X_pca = pca.fit_transform(X_centered)
    # print(total_df.sort_values('CHOL_RISK'))

    # fig, ax = plt.subplots(figsize=(10, 6))
    # plot_pca_embedding(ax, X_pca, total_df['CHOL_RISK'].to_numpy())

    # rf = run_rfc(X_pca, total_df['CHOL_RISK'])
    # feat_imp = plot('cheese', forest=rf, forest_feat=features)
    # feat_imp.plot_importances(n=20)


    # line.scatterplot('LBXTC')
