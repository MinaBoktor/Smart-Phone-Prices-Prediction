import pandas as pd
import numpy as np

# 1. Load files
train_path = r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\\"
test_path = r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\\"

train = pd.read_csv(f"{train_path}train.csv")
test = pd.read_csv(f"{test_path}test.csv")

# 2. FIX: Encode Price to Numbers for Calculation
# We map 'expensive' -> 1 and anything else -> 0 so we can do math.
print("Converting price to numbers for calculation...")
if train['price'].dtype == 'object':
    # Create a temporary numeric column
    train['price_numeric'] = train['price'].apply(lambda x: 1 if str(x).lower() == 'expensive' else 0)
else:
    train['price_numeric'] = train['price']

# 3. Binning Logic
def get_charging_class(watts):
    if watts <= 18:
        return 'Slow'
    elif watts <= 33:
        return 'Standard'
    elif watts <= 67:
        return 'Fast'
    else:
        return 'SuperFast'

# 4. Create Combo
print("Creating Brand-Charging Combo...")
def make_charge_combo(row):
    brand = str(row['brand']).strip()
    charge_class = get_charging_class(row['fast_charging_power'])
    return f"{brand}_{charge_class}"

train['temp_charge_combo'] = train.apply(make_charge_combo, axis=1)
test['temp_charge_combo'] = test.apply(make_charge_combo, axis=1)

# 5. Target Encoding (Using the numeric price)
print("Calculating Average Price Score...")

# Use 'price_numeric' instead of 'price'
charge_target_map = train.groupby('temp_charge_combo')['price_numeric'].mean().to_dict()
global_mean = train['price_numeric'].mean()

train['brand_charging_score'] = train['temp_charge_combo'].map(charge_target_map)
test['brand_charging_score'] = test['temp_charge_combo'].map(charge_target_map).fillna(global_mean)

# 6. Cleanup
# Remove the temporary columns we made
train.drop(columns=['temp_charge_combo', 'price_numeric'], inplace=True, errors='ignore')
test.drop(columns=['temp_charge_combo'], inplace=True, errors='ignore')

# 7. Save
train.to_csv(f"{train_path}train_imputed.csv", index=False)
test.to_csv(f"{test_path}test_imputed.csv", index=False)

print("\nSuccess! 'brand_charging_score' added.")