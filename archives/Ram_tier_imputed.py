import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load the datasets
train_df = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train_imputed.csv")
test_df = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\test_imputed.csv")

# 2. Combine datasets
train_df['is_train'] = 1
test_df['is_train'] = 0
df = pd.concat([train_df, test_df], ignore_index=True)

target = 'RAM Tier'
features = ['RAM Size GB', 'Storage Size GB']

# --- DEBUGGING START ---
print("--- DEBUG INFO ---")
print(f"Unique values in '{target}' before fixing:", df[target].unique())

# 3. Standardize 'Unknown' values
# This converts NaNs, empty strings, or lowercase 'unknown' all into the standard 'Unknown' string
df[target] = df[target].fillna('Unknown')  # Fill actual NaNs
df[target] = df[target].replace(['nan', 'NaN', 'unknown', ''], 'Unknown')

# Check how many rows we actually need to predict
missing_count = df[df[target] == 'Unknown'].shape[0]
print(f"Count of rows marked as 'Unknown' to be imputed: {missing_count}")
print("------------------")

if missing_count == 0:
    print("WARNING: The code found 0 rows to impute. Check your column name or values!")
else:
    # 4. Split data
    known_df = df[df[target] != 'Unknown'].copy()
    unknown_df = df[df[target] == 'Unknown'].copy()

    # 5. Encode Target
    le_target = LabelEncoder()
    y_known = le_target.fit_transform(known_df[target])

    X_known = known_df[features]
    X_unknown = unknown_df[features]

    # 6. Train Random Forest
    # Using more estimators for stability
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_known, y_known)

    # 7. Predict
    y_pred = clf.predict(X_unknown)
    y_pred_labels = le_target.inverse_transform(y_pred)

    # 8. Impute back into main DataFrame
    # We use .loc to ensure the original dataframe is updated
    df.loc[df[target] == 'Unknown', target] = y_pred_labels
    
    print(f"Successfully imputed {len(y_pred_labels)} rows.")

# 9. Split and Save
train_imputed = df[df['is_train'] == 1].drop(columns=['is_train', 'price_encoded'], errors='ignore')
test_imputed = df[df['is_train'] == 0].drop(columns=['is_train', 'price_encoded'], errors='ignore')

train_imputed.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train_imputed.csv", index=False)
test_imputed.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\test_imputed.csv", index=False)

print("\nFinal Check:")
print(f"Remaining 'Unknown' in Train: {train_imputed[target].isin(['Unknown']).sum()}")
print(f"Remaining 'Unknown' in Test: {test_imputed[target].isin(['Unknown']).sum()}")