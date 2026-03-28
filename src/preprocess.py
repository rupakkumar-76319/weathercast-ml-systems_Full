import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

def preprocess_data(df):

    # Missing values
    cat_col = [col for col in df.columns if df[col].dtype == 'object']
    num_col = [col for col in df.columns if df[col].dtype != 'object']

    for col in num_col:
        df[col] = df[col].fillna(df[col].median())

    for col in cat_col:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Drop duplicates
    df = df.drop_duplicates()

    # Date conversion
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop('Date', axis=1, inplace=True)

    # Encode target
    df['RainToday'] = df['RainToday'].map({'Yes': 1, 'No': 0})
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

    # One-hot encoding
    df = pd.get_dummies(df, columns=['Location'], drop_first=True)

    cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
    for col in cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df = pd.get_dummies(df, columns=cols, drop_first=True)

    # Outlier handling
    exclude_cols = ['RainTomorrow', 'RainToday']
    num_col = [col for col in df.select_dtypes(include=np.number).columns if col not in exclude_cols]

    for col in num_col:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.clip(df[col], lower, upper)

    return df