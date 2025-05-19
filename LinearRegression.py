# Tesla Stock Price Prediction System with Proper Time Series Handling
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def display_banner(title):
    """Display a formatted banner for sections"""
    print("\n" + "="*50)
    print(title.center(50))
    print("="*50)

def load_tesla_data():
    """Load and prepare Tesla stock dataset with time series features"""
    try:
        # Load Tesla stock data
        df = pd.read_csv("C:\\Tesla.csv")
        
        # Convert date column to datetime and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Remove potentially leaky features (same-day data)
        leaky_features = ['Open', 'High', 'Low', 'Adj Close']
        df = df.drop(leaky_features, axis=1, errors='ignore')
        
        # Create time-series safe features
        df['Prev_Close'] = df['Close'].shift(1)       # Previous day's close
        df['Prev_Volume'] = df['Volume'].shift(1)     # Previous day's volume
        df['Day_of_Week'] = df.index.dayofweek        # Day of week (0=Monday)
        df['Month'] = df.index.month                  # Month
        
        # Drop rows with missing values from feature creation
        df = df.dropna()
        
        print("\nTesla stock data prepared with time series features")
        print(f"Dataset shape: {df.shape}")
        print(f"Safe features: {[col for col in df.columns if col != 'Close']}")
        
        return df
        
    except Exception as e:
        print(f"\nError loading Tesla data: {e}")
        return None

def preprocess_data(df):
    """Apply standardization to the data with fewer features"""
    # Use only a limited set of features
    features = ['Prev_Close', 'Day_of_Week']  # Fewer features
    X = df[features]
    y = df["Close"]
    
    # Introduce noise to the features
    noise = np.random.normal(0, 5, size=X.shape)  # Add Gaussian noise
    X += noise
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X_scaled_df, y, scaler, features

def evaluate_models(X, y):
    """Evaluate multiple models using time series cross-validation"""
    # Create time series cross-validation splits
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = {
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    display_banner("MODEL EVALUATION")
    print(f"\n{'Model':<20} {'Train R²':<10} {'Test R²':<10} {'Gap':<10}")
    print("-"*50)
    
    for name, model in models.items():
        train_scores, test_scores, gaps = [], [], []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Calculate metrics
            train_r2 = model.score(X_train, y_train)
            y_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            
            # Calculate overfitting gap
            gap = train_r2 - test_r2
            
            train_scores.append(train_r2)
            test_scores.append(test_r2)
            gaps.append(gap)
        
        # Calculate averages
        avg_train = np.mean(train_scores)
        avg_test = np.mean(test_scores)
        avg_gap = np.mean(gaps)
        
        # Print results
        print(f"{name:<20} {avg_train:.4f}    {avg_test:.4f}    {avg_gap:.4f}")
        
        # Store results
        results[name] = {
            'model': model,
            'train_r2': avg_train, 
            'test_r2': avg_test,
            'gap': avg_gap
        }
    
    print("-"*50)
    
    # Return the best model (Linear Regression in this case)
    return results['Linear Regression']['model']

def visualize_predictions(model, X, y):
    """Visualize predictions vs actual values"""
    # Use the last 60 days
    X_vis = X.iloc[-60:]
    y_vis = y.iloc[-60:]
    
    # Make predictions
    y_pred = model.predict(X_vis)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_vis.index, y_vis.values, 'b-', label='Actual Price')
    plt.plot(y_vis.index, y_pred, 'r--', label='Predicted Price')
    plt.title('Tesla Stock Price - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Closing Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Also show scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(y_vis, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.title('Actual vs Predicted Price')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def show_data(df):
    """Display data information"""
    display_banner("TESLA STOCK DATA")
    
    # Show recent data with created features
    print("\nRecent Trading Days with Features:")
    print(df.tail(10))
    
    # Display key statistics
    print("\nKey Statistics:")
    print(df['Close'].describe())
    
    # Show correlation with target
    print("\nFeature Correlation with Close Price:")
    corrs = df.corr()['Close'].sort_values(ascending=False)
    print(corrs)
    
    # Plot closing price history
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'])
    plt.title('Tesla Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Closing Price ($)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def predict_price(model, scaler, features):
    """Make price predictions with proper features"""
    display_banner("TESLA STOCK PRICE PREDICTION")
    
    print("\nEnter yesterday's Tesla stock metrics:")
    
    # Option to use default values
    use_default = input("Use default values? (y/n): ").strip().lower()
    
    if use_default == 'y':
        # Example default values
        inputs = {
            'Prev_Close': 250.0,
            'Day_of_Week': 1  # Tuesday
        }
        
        # Filter to only include available features
        inputs = {f: inputs[f] for f in features if f in inputs}
        print("\nUsing default values:")
        for f, v in inputs.items():
            print(f"{f}: {v}")
        
        # Convert to list in the correct order
        input_values = [inputs.get(f, 0) for f in features]
        
    else:
        # Collect input for each feature
        inputs = []
        for feature in features:
            while True:
                try:
                    val = float(input(f"{feature}: ").strip().replace(',', ''))
                    inputs.append(val)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
        
        input_values = inputs
    
    # Scale input using the same scaler
    input_scaled = scaler.transform([input_values])
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
    X = None
    y = None
    
    while True:
        display_banner("TESLA STOCK PREDICTION SYSTEM")
        print("\nMain Menu:")
        print("1. Load & View Data")
        print("2. Evaluate & Train Models")
        print("3. Visualize Predictions")
        print("4. Predict Closing Price")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            df = load_tesla_data()
            if df is not None:
                show_data(df)
        
        elif choice == '2':
            if df is not None:
                X, y, scaler, features = preprocess_data(df)
                model = evaluate_models(X, y)
                print("\nModel training complete!")
            else:
                print("\nPlease load data first (Option 1)")
        
        elif choice == '3':
            if model is not None and X is not None and y is not None:
                visualize_predictions(model, X, y)
            else:
                print("\nPlease train models first (Option 2)")
        
        elif choice == '4':
            if model and scaler and features:
                predict_price(model, scaler, features)
            else:
                print("\nPlease train model first (Option 2)")
        
        elif choice == '5':
            print("\nThank you for using the Tesla Stock Prediction System!")
            break
        
        else:
            print("\nInvalid choice. Please enter 1-5.")
        
        if choice != '3':  # Skip for visualization option which already waits for plot closing
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()