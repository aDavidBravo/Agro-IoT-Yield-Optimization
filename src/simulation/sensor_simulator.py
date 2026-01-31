import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_sensor_data(n_lots=50, days=365):
    """
    Generates synthetic sensor data for agricultural lots.
    
    Parameters:
        n_lots (int): Number of agricultural lots.
        days (int): Number of days to simulate (default 365).
    """
    np.random.seed(42)
    start_date = datetime.now() - timedelta(days=days)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    data = []
    
    print(f"Generating data for {n_lots} lots over {days} days...")
    
    for lot_id in range(1, n_lots + 1):
        # Base characteristics for the lot (Soil quality variation)
        # Lot 1-10: High quality, Lot 11-30: Medium, Lot 31-50: Variable
        base_quality = np.random.uniform(0.7, 1.0) if lot_id <= 10 else \
                       np.random.uniform(0.5, 0.8) if lot_id <= 30 else \
                       np.random.uniform(0.3, 0.7)
        
        for date in dates:
            # Seasonality for Temperature (Sinusoidal pattern)
            day_of_year = date.timetuple().tm_yday
            seasonality = np.sin((day_of_year - 15) * 2 * np.pi / 365)
            avg_temp = 25 + 5 * seasonality + np.random.normal(0, 2) # Base 25C +/- 5C seasonal +/- noise
            
            # Humidity (inverse to temp roughly, + noise)
            humidity = 60 - 10 * seasonality + np.random.normal(0, 5)
            humidity = np.clip(humidity, 30, 90)
            
            # Soil Sensors
            soil_moisture = humidity * 0.5 + np.random.normal(0, 5) # Correlated with humidity
            soil_moisture = np.clip(soil_moisture, 10, 60)
            
            ph = 6.5 + np.random.normal(0, 0.5) # Neutral-ish pH
            n_level = base_quality * 50 + np.random.normal(0, 5)
            p_level = base_quality * 40 + np.random.normal(0, 5)
            k_level = base_quality * 40 + np.random.normal(0, 5)
            
            # Yield Calculation (The Target)
            # Yield depends on stable temp, good moisture, and high nutrients (base_quality)
            # Optimal temp: 20-30. Deviation reduces yield.
            temp_stress = abs(avg_temp - 25) / 10 # 0 is best
            moisture_stress = abs(soil_moisture - 35) / 25
            
            yield_potential = 100 * base_quality
            actual_yield = yield_potential * (1 - 0.3 * temp_stress - 0.2 * moisture_stress) + np.random.normal(0, 2)
            actual_yield = max(0, actual_yield)
            
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Lot_ID': f'LOT_{lot_id:03d}',
                'Temperature_C': round(avg_temp, 2),
                'Humidity_Pct': round(humidity, 2),
                'Soil_Moisture_Pct': round(soil_moisture, 2),
                'pH': round(ph, 2),
                'Nitrogen_ppm': round(n_level, 2),
                'Phosphorus_ppm': round(p_level, 2),
                'Potassium_ppm': round(k_level, 2),
                'Yield_Ton_Per_Ha': round(actual_yield, 2) # This is usually measured once per season, but we simulate it as a continuous projection for the model
            })

    df = pd.DataFrame(data)
    
    # Save to CSV
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sensor_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}. Shape: {df.shape}")

if __name__ == "__main__":
    generate_sensor_data()
