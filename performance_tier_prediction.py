import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

def impute_performance_tier(df, artifacts=None):

    print("Starting Performance_Tier imputation...")

    work_df = df.copy()

    # 1. Clean Target
    work_df['Performance_Tier_clean'] = work_df['Performance_Tier'].astype(str).str.lower().str.strip()
    unknown_tokens = ['unknown', 'nan', 'none', '']

    # 2. Clean RAM Tier
    work_df['RAM_Tier_clean'] = work_df['RAM Tier'].astype(str).str.lower().str.strip()
    ram_map = {
        'budget': 1, 'mid-range': 2, 'high-end': 3, 'flagship': 4,
        'unknown': 0
    }
    work_df['ram_tier_numeric'] = work_df['RAM_Tier_clean'].map(ram_map).fillna(-1)

    # 3. Clean Processor Series & Encode
    work_df['Processor_Series'] = work_df['Processor_Series'].fillna("unknown").astype(str)

    if artifacts is None:
        le_series = LabelEncoder()
        work_df['series_encoded'] = le_series.fit_transform(work_df['Processor_Series'])
    else:
        le_series = artifacts['encoder']

        known_classes = set(le_series.classes_)
        work_df['Processor_Series'] = work_df['Processor_Series'].apply(lambda x: x if x in known_classes else le_series.classes_[0])
        work_df['series_encoded'] = le_series.transform(work_df['Processor_Series'])

    brand_dummies = pd.get_dummies(work_df['Processor_Brand'], prefix='Brand')

    features = [
        'Clock_Speed_GHz', 'RAM Size GB', 'Core_Count',
        'Refresh_Rate', 'fast_charging_power',
        'Screen_Size', 'Resolution_Width',
        'series_encoded', 'ram_tier_numeric'
    ]

    X = pd.concat([work_df[features], brand_dummies], axis=1)
    X = X.fillna(-1)

    if artifacts is None:
        feature_names = X.columns.tolist()
    else:
        feature_names = artifacts['features']

        X = X.reindex(columns=feature_names, fill_value=0)

    # --- Prediction Logic ---
    y = work_df['Performance_Tier_clean']
    is_unknown = y.isin(unknown_tokens)
    X_unknown = X[is_unknown]

    if artifacts is None:
        X_known = X[~is_unknown]
        y_known = y[~is_unknown]

        hgb = HistGradientBoostingClassifier(max_iter=200, random_state=42)
        hgb.fit(X_known, y_known)

        artifacts = {
            'model': hgb,
            'encoder': le_series,
            'features': feature_names
        }
    else:
        hgb = artifacts['model']

    # Predict if we have unknowns
    if not X_unknown.empty:
        predicted_tiers = hgb.predict(X_unknown)
        df.loc[X_unknown.index, 'Performance_Tier'] = predicted_tiers
        print(f"Imputed {len(predicted_tiers)} values using {'TRAINED' if is_unknown.any() else 'EXISTING'} model.")

    return df, artifacts