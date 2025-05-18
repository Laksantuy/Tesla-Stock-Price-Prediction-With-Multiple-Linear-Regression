# Importing Essential Modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

def load_data():
    """Load and prepare the dataset"""
    # Path to the csv dataset
    datapath = "C:\\Tesla.csv"
    
    try:
        # Loading the csv file
        df = pd.read_csv(datapath)
        
        # Dropping date column from our dataset
        if 'Date' in df.columns:
            df = df.drop(['Date'], axis=1)
            
        return df
    except Exception as e:
        print(f"\nError loading data: {e}")
        return None

def train_model(df):
    """Train the linear regression model"""
    # Creating a shadow copy
    copydf = df.copy()

    # Separating Dependent and Independent Variables
    y = df["Close"]
    x = df.drop(["Close", "Adj Close"], axis=1)

    # Splitting the data into training and testing sets
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)

    # Making the model using linear regression
    model = LinearRegression()
    model.fit(xTrain, yTrain)
    
    return model, xTest, yTest, x.columns

def display_data_info(df):
    """Display information about the dataset"""
    print("\n" + "="*50)
    print("DATASET INFORMATION".center(50))
    print("="*50)
    
    # Printing the first 3 rows of the dataset
    print("\nFirst 3 rows of the dataset:")
    print(df.head(3))

    # Printing statistical information of the dataset
    print("\nStatistical information:")
    print(df.describe())

    # Checking for empty data fields in the dataset
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Checking data types of all columns
    print("\nData types:")
    print(df.info())

def display_model_info(model, xTest, yTest):
    """Display information about the trained model"""
    print("\n" + "="*50)
    print("MODEL INFORMATION".center(50))
    print("="*50)
    
    # Checking the accuracy using R2
    print("\nModel Accuracy (R2 Score):", model.score(xTest, yTest))

    # Getting the intercept and coefficients
    print("\nIntercept:", model.intercept_)
    print("Coefficients:", model.coef_)

def make_prediction(model, feature_columns):
    """Make a prediction based on user input"""
    print("\n" + "="*50)
    print("MAKE PREDICTION".center(50))
    print("="*50)
    
    try:
        # Get all required inputs
        print("\nEnter the following stock details:")
        inputs = []
        for col in feature_columns:
            while True:
                try:
                    val = float(input(f"{col}: ").strip().replace(',', ''))
                    inputs.append(val)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
        
        # Create DataFrame with all inputs
        input_df = pd.DataFrame([inputs], columns=feature_columns)
        print("\nInput DataFrame:")
        print(input_df)
        
        # Make prediction
        prediction = model.predict(input_df)
        print(f"\nPredicted closing price: {prediction[0]:.2f}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

def save_model(model):
    """Save the trained model to a file"""
    try:
        filename = input("\nEnter filename to save model (e.g., 'stock_model.pkl'): ").strip()
        joblib.dump(model, filename)
        print(f"\nModel saved successfully as {filename}")
    except Exception as e:
        print(f"\nError saving model: {e}")

def load_model():
    """Load a trained model from file"""
    try:
        filename = input("\nEnter filename to load model (e.g., 'stock_model.pkl'): ").strip()
        model = joblib.load(filename)
        print("\nModel loaded successfully")
        return model
    except Exception as e:
        print(f"\nError loading model: {e}")
        return None

def main_menu():
    """Display the main menu and handle user choices"""
    model = None
    feature_columns = None
    xTest, yTest = None, None
    df = None
    
    while True:
        print("\n" + "="*50)
        print("STOCK PRICE PREDICTION SYSTEM".center(50))
        print("="*50)
        print("\nMAIN MENU")
        print("1. Load and display dataset")
        print("2. Train model and display information")
        print("3. Make a prediction")
        print("4. Save trained model")
        print("5. Load trained model")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            df = load_data()
            if df is not None:
                display_data_info(df)
        elif choice == '2':
            if df is not None:
                model, xTest, yTest, feature_columns = train_model(df)
                display_model_info(model, xTest, yTest)
            else:
                print("\nPlease load dataset first (Option 1)")
        elif choice == '3':
            if model is not None:
                make_prediction(model, feature_columns)
            else:
                print("\nPlease train or load a model first (Option 2 or 5)")
        elif choice == '4':
            if model is not None:
                save_model(model)
            else:
                print("\nNo model to save. Please train a model first (Option 2)")
        elif choice == '5':
            loaded_model = load_model()
            if loaded_model is not None:
                model = loaded_model
                # Note: When loading a model, you might need to provide feature_columns separately
                # This is a simplified version - in production you'd want to save/load this info too
        elif choice == '6':
            print("\nExiting program. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please enter a number between 1 and 6.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main_menu()