import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def prepare_sample_dataset():
    """
    Create a sample dataset for demonstration purposes.
    This creates a small, manageable dataset for retail forecasting.
    """
    print("🔄 Creating sample dataset...")
    
    # Define products and stores
    products = ['Ramen_Noodles', 'Coffee', 'Bottled_Water', 'Energy_Drink', 'Chips', 
                'Chocolate', 'Cookies', 'Soda', 'Juice', 'Sandwich']
    categories = ['Food', 'Food', 'Food', 'Beverage', 'Snack', 
                  'Snack', 'Snack', 'Beverage', 'Beverage', 'Food']
    stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D']
    
    # Generate dates for 6 months (more data for training)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 30)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Create dataframe
    data = []
    for date in date_range:
        for store in stores:
            for i, product in enumerate(products):
                # Add some randomness to quantities
                base_quantity = np.random.randint(50, 150)
                # Add trend and seasonality
                trend = (date - start_date).days * 0.05
                seasonal = 20 * np.sin(2 * np.pi * date.dayofyear / 365)
                # Weekend effect
                weekend_effect = 30 if date.weekday() >= 5 else 0
                quantity = max(0, int(base_quantity + trend + seasonal + weekend_effect + np.random.normal(0, 10)))
                
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'store': store,
                    'product': product,
                    'sales': quantity,
                    'price': 2.5 if product == 'Ramen_Noodles' else 
                             3.0 if product == 'Coffee' else
                             1.2 if product == 'Bottled_Water' else
                             4.5 if product == 'Energy_Drink' else
                             3.5 if product == 'Chips' else
                             2.0 if product == 'Chocolate' else
                             2.8 if product == 'Cookies' else
                             1.8 if product == 'Soda' else
                             2.2 if product == 'Juice' else
                             4.0,  # Sandwich
                    'category': categories[i]
                })
    
    df = pd.DataFrame(data)
    print(f"📊 Generated dataset with {len(df)} rows")
    return df

def prepare_dataset():
    print("🔄 Loading data...")
    
    # Always create a new sample dataset for better training
    sales = prepare_sample_dataset()
    print("🆕 Created new sample dataset")
    
    print("💾 Saving processed file...")
    os.makedirs("data/processed", exist_ok=True)
    sales.to_csv("data/processed/train.csv", index=False)
    print("✅ Processed dataset saved as data/processed/train.csv")

if __name__ == "__main__":
    prepare_dataset()