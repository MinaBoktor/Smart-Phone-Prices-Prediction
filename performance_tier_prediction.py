import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
train_df = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train.csv")
test_df = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\test.csv")

train_df['is_train'] = 1
test_df['is_train'] = 0
df_full = pd.concat([train_df, test_df], ignore_index=True)

# 2. Preprocessing
ram_map = {'Budget': 0, 'Mid-Range': 1, 'High-End': 2, 'Flagship': 3, 'Unknown': -1}
df_full['ram_tier_numeric'] = df_full['RAM Tier'].map(ram_map)

le_series = LabelEncoder()
df_full['series_encoded'] = le_series.fit_transform(df_full['Processor_Series'].astype(str))

brand_dummies = pd.get_dummies(df_full['Processor_Brand'], prefix='Brand')

features = [
    'Clock_Speed_GHz', 'RAM Size GB', 'Core_Count', 
    'Refresh_Rate', 'fast_charging_power', 
    'Screen_Size', 'Resolution_Width',
    'series_encoded', 'ram_tier_numeric'
]

X = pd.concat([df_full[features], brand_dummies], axis=1)
y = df_full['Performance_Tier']

# 3. Train Model (No Price)
X_known = X[y != 'Unknown']
y_known = y[y != 'Unknown']
X_unknown = X[y == 'Unknown']

gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb.fit(X_known, y_known)

# 4. Predict ALL Unknowns (No Threshold)
# We trust the model's prediction more than "Unknown" (0)
predicted_tiers = gb.predict(X_unknown)

# 5. Fill and Save
df_full.loc[df_full['Performance_Tier'] == 'Unknown', 'Performance_Tier'] = predicted_tiers

train_imputed = df_full[df_full['is_train'] == 1].drop(['is_train', 'ram_tier_numeric', 'series_encoded'], axis=1)
test_imputed = df_full[df_full['is_train'] == 0].drop(['is_train', 'ram_tier_numeric', 'series_encoded'], axis=1)

train_imputed.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train_imputed.csv", index=False)
test_imputed.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\test_imputed.csv", index=False)
print("Files saved. All 'Unknown' values have been replaced.")