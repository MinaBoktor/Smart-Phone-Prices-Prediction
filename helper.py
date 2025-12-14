import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt







def preprocess(df, train=True, training_columns=[], imputation_values=None, scaler=None):

    #Binaray_columns_versus_price_(df)
    #unique(df)

    # Price Binary Encoding
    if "price" in df.columns:
            df["price"] = df["price"].map({'expensive': 1, 'non-expensive': 0})



    numerical_columns=['rating','Processor_Series','Core_Count', 'Clock_Speed_GHz','RAM Size GB','Storage Size GB','battery_capacity', 'fast_charging_power' , 'Screen_Size','Resolution_Width','Resolution_Height','Refresh_Rate','primary_rear_camera_mp','num_rear_cameras','primary_front_camera_mp','num_front_cameras']

    if train:
        imputation_values = {}
    elif imputation_values is None:
        raise ValueError("imputation_values must be provided when train=False!")

    for y in numerical_columns:
        df[y] = pd.to_numeric(df[y], errors="coerce")

        if train:
            mean_value = df[y].mean()
            imputation_values[y] = mean_value
        else:
            mean_value = imputation_values.get(y, 0)

        df[y] = df[y].fillna(mean_value)


    df = df.apply(lambda col: col.str.lower() if col.dtype == "object" else col)

    # Manual Binary Encoding
    manual_encoding = ["NFC", "memory_card_support", "5G", ]
    for y in manual_encoding:
        df[f"{y}_encoded"] = df[y].map({'yes': 1, 'no': 0})


    # Split os_version into major, minor, patch
    df["os_version"] = df["os_version"].str.replace("v", "")

    split_cols = df["os_version"].str.split('.', n=2, expand=True)

    if 2 not in split_cols.columns:
        split_cols[2] = 0

    if 1 not in split_cols.columns:
        split_cols[1] = 0

    df["os_version_major"] = split_cols[0].astype(int)
    df["os_version_minor"] = split_cols[1].fillna(0).astype(int)
    df["os_version_patch"] = split_cols[2].fillna(0).astype(int)

    # Ordinal Encoding for RAM Tier
    encoder = OrdinalEncoder(categories=[
        ["unknown", "budget", "mid-range", "high-end", "flagship"],
    ])

    df["RAM Tier_encoded"] = encoder.fit_transform(df[["RAM Tier"]])


    # Converting into GBs in memory_card_size
    def convert_storage(x):
        x = x.lower().strip()
        if "tb" in x:
            return float(x.replace("tb", "")) * 1024
        if "gb" in x:
            return float(x.replace("gb", ""))
        return None

    df["memory_card_size"] = df["memory_card_size"].apply(convert_storage)


    # Handling unrealistic inputs by setting lower limits
    lower_limits = {
        'RAM Size GB': 0.5,
        'Storage Size GB': 4,
        }

    for col, limit in lower_limits.items():
        df[col] = df[col].clip(lower=limit, upper=None)

    upper_limits = {
        'battery_capacity': 10000,
    }

    for col, limit in upper_limits.items():
        df[col] = df[col].clip(lower=None, upper=limit)



    # Outlier Capping for numerical columns using IQR method
    cols_to_cap = ['rating', 'Processor_Series']

    for col in cols_to_cap:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1


        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"Capping outliers in {col}: Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")

        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    #df.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\preproccessed_outliers.csv", index=False)

    

    # Standard Scaling
    cols_to_scale = numerical_columns + ["memory_card_size", "os_version_major", "os_version_minor", "os_version_patch", "RAM Tier_encoded"]
    
    if train:
        # LEARN: Create new scaler and fit it on training data
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        # APPLY: Use the scaler learned from training (Safety check)
        if scaler is None:
             raise ValueError("Scaler must be provided when train=False!")
        
        # Transform test data using the existing scaler's stats
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    #analysis(df)

    # One Hot Key Encoding (The condition train indicates whether we are processing training data or test data)
    one_hot_key = ["Processor_Brand", "Notch_Type", "brand"]
    df = pd.get_dummies(df, columns=one_hot_key, dtype=int)
    if train:
        training_columns = df.columns.tolist()
    else:
        if training_columns is None:
            raise ValueError("training_cols must be provided when is_training=False.")

        df = df.reindex(columns=training_columns, fill_value=0)


    # Dropping Unnecessary Columns which were encoded and unrelated features
    unrelated_features = ["rating", "Processor_Series", "os_name", "Dual_Sim", "Vo5G", "num_rear_cameras", "os_version_patch", "Core_Count", "os_version_minor", "IR_Blaster", "4G",  ]
    df = df.drop(manual_encoding+unrelated_features+["RAM Tier", "os_version", "memory_card_size", "Performance_Tier"], axis=1)


    return df, training_columns, imputation_values, scaler


def Binaray_columns_versus_price_(df):
    target_col = 'price'
    feature_cols = ['Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'memory_card_support', 'IR_Blaster']

    # Print a Header for readability
    print(f"{'Feature':<15} | {'Yes -> Expensive':<18} | {'No -> Non-Expensive':<18}")
    print("-" * 60)

    for feature in feature_cols:
        # Calculate percentages
        ct = pd.crosstab(df[feature], df[target_col], normalize='index')

        # Extract values safely (returns 0 if specific key is missing)
        yes_score = ct.loc['Yes', 'expensive'] if 'Yes' in ct.index else 0
        no_score = ct.loc['No', 'non-expensive'] if 'No' in ct.index else 0

        # Print immediately
        print(f"{feature:<15} | {yes_score:.1%}              | {no_score:.1%}")

def unique(df):
    # Set pandas to display the full list without truncating (optional but helpful)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)

    for col in df.columns:
        print(f"--- Column: {col} ---")
        # This prints the full array of unique values
        print(df[col].unique()) 
        print("\n") # Adds a blank line for readability

def analysis(df):

    unknown_percentages = (df.eq("Unknown").sum() / len(df)) * 100
    print(unknown_percentages)

    categorical_cols = df.select_dtypes(include=['object'])
    for col in categorical_cols:
        top_freq = df[col].value_counts(normalize=True).iloc[0]
        variability = 1 - top_freq
        print(f"{col}: {variability:.4f}")

        print("\n--- Correlation with Price ---")
            
        # Calculate correlation (numeric_only=True prevents errors with any remaining text)
        correlation = df.corrwith(df['price'], numeric_only=True).sort_values(ascending=False)
        
        # Remove 'price' itself from the list
        if 'price' in correlation:
            correlation = correlation.drop('price')
        
        sorted_correlation = correlation.iloc[correlation.abs().argsort()]

        # 4. Print
        with pd.option_context('display.max_rows', None):
            print(sorted_correlation)


if __name__ == "__main__":
    df=pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train.csv")
    df, _, _, _ = preprocess(df, train=True)
    df.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\preproccessed_train.csv", index=False)
