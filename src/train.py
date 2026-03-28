import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_model():

    # Load data
    train_df= pd.read_csv(r"M:\ML_Project\weathercast-ml-system\data\processed\train.csv")
    val_df= pd.read_csv(r"M:\ML_Project\weathercast-ml-system\data\processed\validation.csv")

    # Split features and target
    X_train= train_df.drop('RainTomorrow', axis=1)
    y_train= train_df['RainTomorrow']

    X_val= val_df.drop('RainTomorrow', axis=1)
    y_val= val_df['RainTomorrow']

    # ✅ SAVE COLUMN NAMES (ADD THIS LINE)
    os.makedirs('models', exist_ok=True)
    joblib.dump(X_train.columns.tolist(), 'models/columns.pkl')

    # STEP 1: SCALING
    scaler= StandardScaler()
    X_train= scaler.fit_transform(X_train)
    X_val= scaler.transform(X_val)

    # STEP 2: TRAIN MODEL
    model= LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # STEP 3: EVALUATION
    y_pred= model.predict(X_val)

    print("Accuracy: ", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    # STEP 4: SAVE MODEL
    joblib.dump(model,'models/model.pkl')
    joblib.dump(scaler,'models/scaler.pkl')

    print('Model, Scaler, and Columns are saved')

if __name__=='__main__':
    train_model()