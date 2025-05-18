# Tesla Stock Price Prediction System
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def display_banner(title):
    """Display a formatted banner for sections"""
    print("\n" + "="*50)
    print(title.center(50))
    print("="*50)

def load_tesla_data():
    """Load and prepare Tesla stock dataset"""
    try:
        # Load Tesla stock data
        df = pd.read_csv("C:\\Tesla.csv")
        
        # Display loading message
        print("\nLoading Tesla stock data...")
        print(f"Found {len(df)} days of trading data")
        
        # Drop date column if it exists
        if 'Date' in df.columns:
            df = df.drop(['Date'], axis=1)
            
        return df
    except Exception as e:
        print(f"\nError loading Tesla data: {e}")
        return None

def train_tesla_model(df):
    """Train the Tesla stock prediction model"""
    print("\nTraining Tesla stock prediction model...")
    
    # Separate Dependent and Independent Variables
    y = df["Close"]  # We're predicting closing price
    x = df.drop(["Close", "Adj Close"], axis=1)  # Using other metrics as predictors
    
    # Split the data (80% training, 20% testing)
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2)
    
    # Create and train linear regression model
    model = LinearRegression()
    model.fit(xTrain, yTrain)
    
    # Calculate accuracy
    accuracy = model.score(xTest, yTest)
    
    return model, x.columns, accuracy

def show_tesla_data(df):
    """Display Tesla stock data information"""
    display_banner("TESLA STOCK DATA")
    
    print("\nRecent Trading Days:")
    print(df.head(10))
    
    print("\nKey Statistics:")
    print(df.describe())
    
    print("\nMissing Values Check:")
    print(df.isnull().sum())

def predict_tesla_price(model, features):
    """Predict Tesla stock closing price based on user input"""
    display_banner("TESLA STOCK PRICE PREDICTION")
    
    print("\nEnter Tesla stock metrics for prediction:")
    print("(Typical values: Open=250.00, High=255.00, Low=245.00, Volume=15000000)")
    
    inputs = []
    for col in features:
        while True:
            try:
                # Get user input for each stock metric
                val = float(input(f"{col}: ").strip().replace(',', ''))
                inputs.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    
    # Create DataFrame with user inputs
    input_df = pd.DataFrame([inputs], columns=features)
    
    # Make prediction
    prediction = model.predict(input_df)
    
    print("\nPrediction Results:")
    print("-"*40)
    print("Input Metrics:")
    print(input_df)
    print(f"\nPredicted Tesla Closing Price: ${prediction[0]:.2f}")
    print("-"*40)

def main():
    """Main Tesla Stock Prediction System"""
    # Initialize variables
    tesla_model = None
    model_features = None
    model_accuracy = None
    tesla_data = None
    
    while True:
        display_banner("TESLA STOCK PREDICTION SYSTEM")
        print("\nMain Menu:")
        print("1. Load & View Tesla Stock Data")
        print("2. Train Prediction Model")
        print("3. Predict Closing Price")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            tesla_data = load_tesla_data()
            if tesla_data is not None:
                show_tesla_data(tesla_data)
        elif choice == '2':
            if tesla_data is not None:
                tesla_model, model_features, model_accuracy = train_tesla_model(tesla_data)
                print(f"\nTesla prediction model trained successfully!")
                print(f"Model Accuracy (R² Score): {model_accuracy:.4f}")
            else:
                print("\nPlease load Tesla data first (Option 1)")
        elif choice == '3':
            if tesla_model is not None:
                predict_tesla_price(tesla_model, model_features)
            else:
                print("\nPlease train the model first (Option 2)")
        elif choice == '4':
            print("\nThank you for using the Tesla Stock Prediction System!")
            break
        else:
            print("\nInvalid choice. Please enter 1-4.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()