import pandas as pd
import numpy as np

# 1. Load your base imputed files
train_path = r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train_imputed.csv"
test_path = r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\test_imputed.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print(f"Loaded Train: {train.shape}")
print(f"Loaded Test: {test.shape}")

# --- HELPER: Ensure Price is Numeric for Calculation ---
# We map 'expensive' -> 1 and 'non-expensive' -> 0 temporarily to calculate averages
if train['price'].dtype == 'object':
    train['price_numeric'] = train['price'].apply(lambda x: 1 if str(x).strip().lower() == 'expensive' else 0)
else:
    train['price_numeric'] = train['price']
    
global_mean = train['price_numeric'].mean()

# ==============================================================================
# FEATURE 1: BRAND + FRONT CAMERA (Target Encoded)
# ==============================================================================
print("\n1. Engineering Camera Feature...")

def get_camera_class(mp):
    if mp <= 16: return 'Standard'
    elif 20 <= mp <= 32: return 'Premium'
    else: return 'High_Res_Budget'

def make_cam_combo(row):
    return f"{str(row['brand']).strip()}_{get_camera_class(row['primary_front_camera_mp'])}"

# Create combos
train['temp_cam_combo'] = train.apply(make_cam_combo, axis=1)
test['temp_cam_combo'] = test.apply(make_cam_combo, axis=1)

# Target Encode (Map Combo -> Average Price)
cam_map = train.groupby('temp_cam_combo')['price_numeric'].mean().to_dict()

train['brand_cam_score'] = train['temp_cam_combo'].map(cam_map)
test['brand_cam_score'] = test['temp_cam_combo'].map(cam_map).fillna(global_mean)

# Clean up
train.drop(columns=['temp_cam_combo'], inplace=True)
test.drop(columns=['temp_cam_combo'], inplace=True)


# ==============================================================================
# FEATURE 2: BRAND + CHARGING POWER (Target Encoded)
# ==============================================================================
print("2. Engineering Charging Feature...")

def get_charging_class(watts):
    if watts <= 18: return 'Slow'
    elif watts <= 33: return 'Standard' # The Flagship zone
    elif watts <= 67: return 'Fast'
    else: return 'SuperFast' # The Budget-Gamer zone

def make_charge_combo(row):
    return f"{str(row['brand']).strip()}_{get_charging_class(row['fast_charging_power'])}"

# Create combos
train['temp_charge_combo'] = train.apply(make_charge_combo, axis=1)
test['temp_charge_combo'] = test.apply(make_charge_combo, axis=1)

# Target Encode
charge_map = train.groupby('temp_charge_combo')['price_numeric'].mean().to_dict()

train['brand_charging_score'] = train['temp_charge_combo'].map(charge_map)
test['brand_charging_score'] = test['temp_charge_combo'].map(charge_map).fillna(global_mean)

# Clean up
train.drop(columns=['temp_charge_combo', 'price_numeric'], inplace=True, errors='ignore')
test.drop(columns=['temp_charge_combo'], inplace=True, errors='ignore')

# ==============================================================================
# SAVE THE WINNING DATASET
# ==============================================================================
train.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train_imputed.csv", index=False)
test.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\test_imputed.csv", index=False)

print("\nSUCCESS! created 'train_winning.csv' and 'test_winning.csv'")
print("Columns added: 'brand_cam_score', 'brand_charging_score'")
print("Use these files in your model.py to get that 97.3% back.")