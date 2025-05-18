# Tesla Stock Price Prediction System with Standardization
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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
        
        # Simply drop the Date column if it exists
        if 'Date' in df.columns:
            df = df.drop(['Date'], axis=1)
        
        print("\nTesla stock data loaded successfully")
        print(f"Dataset shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"\nError loading Tesla data: {e}")
        return None

def preprocess_data(df):
    """Apply standardization to the data"""
    X = df.drop(["Close", "Adj Close"], axis=1)
    y = df["Close"]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X, y, scaler

def train_model(X, y):
    """Train the prediction model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    return model, accuracy

def show_data(df):
    """Display data information"""
    display_banner("TESLA STOCK DATA")
    print("\nRecent Trading Days:")
    print(df.head(3))
    print("\nKey Statistics:")
    print(df.describe())

def predict_price(model, scaler, features):
    """Make price predictions"""
    display_banner("TESLA STOCK PRICE PREDICTION")
    
    print("\nEnter Tesla stock metrics:")
    print("(Example: Open=250.00, High=255.00, Low=245.00, Volume=15000000)")
    
    inputs = []
    for feature in features:
        while True:
            try:
                val = float(input(f"{feature}: ").strip().replace(',', ''))
                inputs.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
    
    # Scale input using the same scaler
    input_scaled = scaler.transform([inputs])
    prediction = model.predict(input_scaled)
    
    print("\nPrediction Results:")
    print("-"*40)
    print(f"Predicted Tesla Closing Price: ${prediction[0]:.2f}")
    print("-"*40)

def main():
    """Main application flow"""
    model = None
    scaler = None
    features = None
    df = None
    
    while True:
        display_banner("TESLA STOCK PREDICTION SYSTEM")
        print("\nMain Menu:")
        print("1. Load & View Data")
        print("2. Preprocess & Train Model")
        print("3. Predict Closing Price")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            df = load_tesla_data()
            if df is not None:
                show_data(df)
                features = df.drop(["Close", "Adj Close"], axis=1).columns.tolist()
        
        elif choice == '2':
            if df is not None:
                X, y, scaler = preprocess_data(df)
                model, accuracy = train_model(X, y)
                print(f"\nModel trained successfully! Accuracy (R²): {accuracy:.4f}")
            else:
                print("\nPlease load data first (Option 1)")
        
        elif choice == '3':
            if model and scaler and features:
                predict_price(model, scaler, features)
            else:
                print("\nPlease train model first (Option 2)")
        
        elif choice == '4':
            print("\nThank you for using the Tesla Stock Prediction System!")
            break
        
        else:
            print("\nInvalid choice. Please enter 1-4.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()