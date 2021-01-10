import pandas as pd
import numpy as np


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
    nan_nums = []  # in parrallel with nan_columns
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
