from data_loader import load_data
from preprocess import preprocess_data
from split import split_data
from train import train_model
from evaluate import evaluate_model
from predict import predict
import pandas as pd
import os

def run_pipeline():

    file_path = r'M:\ML_Project\weathercast-ml-system\data\raw\weatherAUS.csv'

    # Step 1: Load
    df = load_data(file_path)

    # Step 2: Preprocess
    df = preprocess_data(df)

    # Step 3: Split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Step 4: Save
    base_path = r'M:\ML_Project\weathercast-ml-system\data\processed'
    os.makedirs(base_path, exist_ok=True)

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(base_path + r'\train.csv', index=False)
    val_df.to_csv(base_path + r'\validation.csv', index=False)
    test_df.to_csv(base_path + r'\test.csv', index=False)

    print("Machine Learning pipeline completed successfully!")

    # Train the model 
    train_model()

    # Test the model
    evaluate_model()

    # # Predict the model
    # input_data= pd.read_csv(r"M:\ML_Project\weathercast-ml-system\data\processed\test.csv")
    # print("Test data Summary: ")
    # predict(input_data)

if __name__ == "__main__":
    run_pipeline()