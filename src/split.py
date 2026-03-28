from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df):

    X = df.drop('RainTomorrow', axis=1)
    y = df['RainTomorrow']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test