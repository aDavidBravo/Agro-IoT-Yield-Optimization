import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import pickle

def train_yield_model():
    """
    Trains a Random Forest model to predict crop yield and calculates estimated savings.
    """
    data_path = 'data/raw/sensor_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Run sensor_simulator.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Feature Engineering: Aggregate by Lot_ID for a season-level prediction
    # We assume the dataset represents one season of data.
    lot_stats = df.groupby('Lot_ID').agg({
        'Temperature_C': 'mean',
        'Humidity_Pct': 'mean',
        'Soil_Moisture_Pct': 'mean',
        'pH': 'mean',
        'Nitrogen_ppm': 'mean',
        'Phosphorus_ppm': 'mean',
        'Potassium_ppm': 'mean',
        'Yield_Ton_Per_Ha': 'mean' # Target
    }).reset_index()
    
    X = lot_stats[['Temperature_C', 'Humidity_Pct', 'Soil_Moisture_Pct', 'pH', 'Nitrogen_ppm', 'Phosphorus_ppm', 'Potassium_ppm']]
    y = lot_stats['Yield_Ton_Per_Ha']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training Random Forest on {len(X_train)} lots...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\n--- Model Performance ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Business Metric: Estimated Savings
    # Baseline Cost per Lot (Assumption): $1000 (Water, Fertilizer, Labor) for the season
    # Potential Optimization: 12% (based on user context)
    # Realized Savings = Baseline * 12% * Model_Confidence (R2)
    
    baseline_cost_per_lot = 1000
    total_lots = len(lot_stats)
    max_savings_potential = total_lots * baseline_cost_per_lot * 0.12
    
    # We use R2 as a proxy for confidence/capture rate of the variance
    realized_savings = max_savings_potential * max(0, r2)
    
    print("\n--- Business Impact (Agro-Industrial Leader) ---")
    print(f"Total Lots Analyzed: {total_lots}")
    print(f"Max Potential Savings (12% Efficiency): ${max_savings_potential:,.2f}")
    print(f"Estimated Realized Savings (scaled by accuracy): ${realized_savings:,.2f}")
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    with open('models/yield_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModel saved to models/yield_model.pkl")

if __name__ == "__main__":
    train_yield_model()
