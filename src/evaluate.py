# Calculates MAE, RMSE, R²

import pandas as pd
import joblib 
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model():
    
    # Load Data
    test_df= pd.read_csv(r'M:\ML_Project\weathercast-ml-system\data\processed\test.csv')

    # Split feature and Target
    X_test= test_df.drop('RainTomorrow', axis=1)
    y_test= test_df['RainTomorrow']

    # Load the train or validate model
    model= joblib.load('models/model.pkl')
    scaler= joblib.load('models/scaler.pkl')

    # Scaling the test data
    X_test= scaler.transform(X_test)

    # Predict data
    y_pred= model.predict(X_test)

    print("Test Accuracy: ",accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__== "__main__":
    evaluate_model()
