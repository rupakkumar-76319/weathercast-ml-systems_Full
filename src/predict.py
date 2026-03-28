import joblib
import pandas as pd
import numpy as np
import os

def predict(input_data):

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Load model
    model = joblib.load(os.path.join(BASE_DIR, "models/model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # ---- SAME PREPROCESSING AS TRAIN ----

    # Create cyclical features
    df['month_sin'] = np.sin(2*np.pi*df['Month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['Month']/12)

    # IMPORTANT: match training columns
    train_cols = joblib.load('models/columns.pkl')  # we will create this

    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=train_cols, fill_value=0)

    # Scaling (if used)
    if scaler is not None:
        df = scaler.transform(df)

    # Prediction
    prediction = model.predict(df)

    return prediction[0]