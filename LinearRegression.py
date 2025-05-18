# Tesla Stock Price Prediction System (Improved Version)
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt

def load_tesla_data():
    """Load Tesla data with proper time-series features (no leakage)"""
    try:
        # Load data
        df = pd.read_csv("C:\\Tesla.csv")
        
        # Convert date and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Remove all potentially leaky features
        leaky_features = ['Open', 'High', 'Low', 'Adj Close']
        df = df.drop(leaky_features, axis=1, errors='ignore')
        
        # Create time-series safe features
        df['Prev_Close'] = df['Close'].shift(1)  # Yesterday's closing price
        df['Prev_Volume'] = df['Volume'].shift(1)  # Yesterday's volume
        df['Day_of_Week'] = df.index.dayofweek    # Monday=0, Sunday=6
        
        # Add more lag features (previous days' prices) to capture trends
        df['Prev_Close_2'] = df['Close'].shift(2)  # 2 days ago closing price
        df['Prev_Close_3'] = df['Close'].shift(3)  # 3 days ago closing price
        df['Prev_Close_5'] = df['Close'].shift(5)  # 5 days ago closing price
        
        # Weekly moving averages (with proper lag to avoid leakage)
        df['MA_5'] = df['Close'].shift(1).rolling(window=5).mean()  # 5-day moving average
        
        # Price changes and volatility features
        df['Price_Change_1d'] = df['Prev_Close'] - df['Prev_Close_2']  # 1-day price change
        df['Price_Change_Pct_1d'] = (df['Prev_Close'] / df['Prev_Close_2'] - 1) * 100  # 1-day percent change
        
        # Volume changes
        df['Volume_Change_1d'] = df['Prev_Volume'] - df['Volume'].shift(2)
        df['Volume_Change_Pct_1d'] = (df['Prev_Volume'] / df['Volume'].shift(2) - 1) * 100
        
        df = df.dropna()  # Remove rows with missing values
        
        print("\nSafe features:", df.columns.tolist())
        return df
        
    except Exception as e:
        print(f"\nError loading data: {e}")
        return None

def preprocess_data(df):
    """Standardize features without leakage - using fewer features"""
    # Select only the most essential features
    # This is a reduced set to avoid potential overfitting
    essential_features = [
        'Prev_Close',      # Yesterday's closing price (most important predictor)
        'Prev_Volume',     # Yesterday's volume
        'Day_of_Week',     # Calendar effect
        'MA_5'             # 5-day moving average (trend indicator)
    ]
    
    # Filter to only include available columns
    feature_cols = [col for col in essential_features if col in df.columns]
    print(f"\nUsing reduced feature set: {feature_cols}")
    
    X = df[feature_cols]
    y = df['Close']  # Target: Today's closing price
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models to compare performance"""
    tscv = TimeSeriesSplit(n_splits=5, test_size=30)  # Larger test size
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge (alpha=1.0)': Ridge(alpha=1.0),
        'Ridge (alpha=5.0)': Ridge(alpha=5.0),
        'Ridge (alpha=20.0)': Ridge(alpha=20.0),
    }
    
    results = {}
    
    # Baseline model for comparison
    dummy = DummyRegressor(strategy="mean")
    dummy_scores = []
    
    # Plot setup for later visualization
    plt.figure(figsize=(15, 10))
    
    print("\n" + "="*70)
    print(f"{'Model':<20} {'Train R²':<10} {'Test R²':<10} {'Gap':<10} {'MAE':<10}")
    print("="*70)
    
    # Last fold for visualization
    last_split = list(tscv.split(X))[-1]
    train_idx, test_idx = last_split
    X_train_last, X_test_last = X.iloc[train_idx], X.iloc[test_idx]
    y_train_last, y_test_last = y.iloc[train_idx], y.iloc[test_idx]
    
    subplot_idx = 1
    
    for name, model in models.items():
        train_scores, test_scores, mae_scores, gaps = [], [], [], []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Calculate metrics
            train_r2 = model.score(X_train, y_train)
            y_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate overfitting gap
            gap = train_r2 - test_r2
            
            train_scores.append(train_r2)
            test_scores.append(test_r2)
            mae_scores.append(mae)
            gaps.append(gap)
        
        # Dummy model on last fold
        if name == 'Linear Regression':
            dummy.fit(X_train, y_train)
            dummy_pred = dummy.predict(X_test)
            dummy_r2 = r2_score(y_test, dummy_pred)
            dummy_scores.append(dummy_r2)
        
        # Calculate averages
        avg_train = np.mean(train_scores)
        avg_test = np.mean(test_scores)
        avg_gap = np.mean(gaps)
        avg_mae = np.mean(mae_scores)
        
        # Print results
        print(f"{name:<20} {avg_train:.4f}    {avg_test:.4f}    {avg_gap:.4f}    ${avg_mae:.2f}")
        
        # Store results
        results[name] = {
            'model': model,
            'train_r2': avg_train,
            'test_r2': avg_test,
            'gap': avg_gap,
            'mae': avg_mae
        }
        
        # Plot for this model
        plt.subplot(2, 2, subplot_idx)
        model.fit(X_train_last, y_train_last)
        y_pred_last = model.predict(X_test_last)
        
        plt.scatter(y_test_last, y_pred_last, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        plt.title(f"{name}: R² = {test_scores[-1]:.4f}, Gap = {gaps[-1]:.4f}")
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        subplot_idx += 1
    
    print("-"*70)
    print(f"Baseline (Mean)         {'N/A':<10} {np.mean(dummy_scores):.4f}")
    print("="*70)
    
    # Find best model (highest test R² with acceptable gap)
    best_model = None
    best_score = -np.inf
    
    for name, metrics in results.items():
        # Prefer models with good test performance and small gap
        if metrics['test_r2'] > best_score and metrics['gap'] < 0.3:
            best_score = metrics['test_r2']
            best_model = name
    
    if best_model:
        print(f"\nRecommended model: {best_model}")
        print(f"R² = {results[best_model]['test_r2']:.4f}, Overfitting gap = {results[best_model]['gap']:.4f}")
        
        # Interpret results in stock prediction context
        if results[best_model]['test_r2'] > 0.6:
            print("\nNOTE: R² values for stock price prediction:")
            print("- R² > 0.7: Unusually high for stock prediction, may indicate data leakage")
            print("- R² 0.4-0.6: Good performance for stock price models")
            print("- R² 0.2-0.4: Typical for most stock prediction models")
            print("- R² < 0.2: Poor performance, model has little predictive power")
            
            if results[best_model]['test_r2'] > 0.7:
                print("\nYour model is performing at an unusually high level for stock prediction.")
                print("Consider whether there might be:")
                print("1. Temporal leakage (using future data to predict past)")
                print("2. Unrealistic test/train splits")
                print("3. Unusual patterns in this particular stock")
                print("4. Features that indirectly leak target information")
        else:
            print("\nModel performance appears realistic for stock price predictions.")
    
    plt.tight_layout()
    plt.show()
    
    # Return the actual model object that performed best
    return results[best_model]['model'] if best_model else results['Ridge (alpha=1.0)']['model']

def feature_importance(model, feature_names):
    """Analyze feature importance if model supports it"""
    if hasattr(model, 'coef_'):
        # Get absolute coefficients
        coefs = np.abs(model.coef_)
        # Create mapping of feature names to their importance
        importance = dict(zip(feature_names, coefs))
        # Sort by importance
        sorted_importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
        
        print("\nFeature Importance:")
        for feature, importance in sorted_importance.items():
            print(f"{feature:<20} {importance:.4f}")
    else:
        print("\nModel doesn't support coefficient inspection for feature importance.")

def predict_price(model, scaler, feature_names):
    """Make predictions with user input"""
    print("\nEnter required features for prediction:")
    print("(Enter 'default' to use example values)")
    
    # Default values example
    default_values = {
        'Prev_Close': 250.0,
        'Prev_Volume': 20000000,
        'Day_of_Week': 1,
        'Prev_Close_2': 245.0,
        'Prev_Close_3': 248.0,
        'Prev_Close_5': 240.0,
        'MA_5': 247.0,
        'Price_Change_1d': 5.0,
        'Price_Change_Pct_1d': 2.0,
        'Volume_Change_1d': 2000000,
        'Volume_Change_Pct_1d': 10.0
    }
    
    choice = input("Use default values? (yes/no): ").strip().lower()
    
    if choice == 'yes' or choice == 'default' or choice == 'y':
        inputs = [default_values.get(feature, 0) for feature in feature_names]
    else:
        inputs = []
        for feature in feature_names:
            while True:
                try:
                    val = float(input(f"{feature}: "))
                    inputs.append(val)
                    break
                except ValueError:
                    print("Invalid input. Please enter a numeric value.")
    
    # Scale input and predict
    input_df = pd.DataFrame([inputs], columns=feature_names)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    
    print("\n" + "="*50)
    print(f"Predicted Tesla Closing Price: ${prediction[0]:.2f}")
    print("="*50)

def main():
    """Main program execution"""
    print("\nTesla Stock Price Prediction System (Reduced Feature Version)")
    print("This version uses fewer features to reduce overfitting")
    
    df = load_tesla_data()
    if df is None:
        return
    
    X, y, scaler = preprocess_data(df)
    feature_names = X.columns.tolist()
    
    # Train and evaluate multiple models
    best_model = train_and_evaluate_models(X, y)
    
    # Analyze feature importance
    feature_importance(best_model, feature_names)
    
    # Prediction loop
    while True:
        print("\nOptions:")
        print("1. Make new prediction")
        print("2. Exit")
        choice = input("Enter choice: ").strip()
        
        if choice == '1':
            predict_price(best_model, scaler, feature_names)
        elif choice == '2':
            print("Exiting program.")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()