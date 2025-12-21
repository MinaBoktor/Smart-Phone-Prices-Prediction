import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from performance_tier_prediction import impute_performance_tier


def preprocess(df, train=True, training_columns=[], imputation_values=None, scaler=None, imputer_artifacts=None):

    #Visual_Binary_columns_versus_price(df)
    #unique(df)
    #analyze_unknown_percentage(df)


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

    df, imputer_artifacts = impute_performance_tier(df, artifacts=imputer_artifacts)

    # Manual Binary Encoding
    manual_encoding = ["NFC", "memory_card_support", "5G", "IR_Blaster", "4G", "Dual_Sim", "Vo5G"]
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
    ram_encoder = OrdinalEncoder(categories=[
        ["unknown", "budget", "mid-range", "high-end", "flagship"],
    ])

    tier_encoder = OrdinalEncoder(categories=[
        ["unknown", "budget", "mid-range", "high-end", "flagship"],
    ])

    df["RAM Tier_encoded"] = ram_encoder.fit_transform(df[["RAM Tier"]])
    df["Performance_Tier_encoded"] = tier_encoder.fit_transform(df[["Performance_Tier"]])


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


    # Standard Scaling
    cols_to_scale = numerical_columns + ["memory_card_size", "os_version_major", "os_version_minor", "os_version_patch", "RAM Tier_encoded", "Performance_Tier_encoded", "primary_front_camera_mp"]

    if train:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        if scaler is None:
             raise ValueError("Scaler must be provided when train=False!")
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Analyze Data

    #analyze_categorical_variability(df)

    #analyze_price_correlation(df)

    # One Hot Key Encoding (The condition train indicates whether we are processing training data or test data)
    one_hot_key = ["Processor_Brand", "Notch_Type", "brand", "os_name"]
    df = pd.get_dummies(df, columns=one_hot_key, dtype=int)
    if train:
        training_columns = df.columns.tolist()
    else:
        if training_columns is None:
            raise ValueError("training_cols must be provided when is_training=False.")

        df = df.reindex(columns=training_columns, fill_value=0)


    # Dropping Unnecessary Columns which were encoded and unrelated features
    unrelated_features = []
    df = df.drop(manual_encoding+unrelated_features+["RAM Tier", "os_version", "memory_card_size", "Performance_Tier"], axis=1)


    return df, training_columns, imputation_values, scaler, imputer_artifacts


def Visual_Binary_columns_versus_price(df):
    target_col = 'price'
    feature_cols = ['Dual_Sim', '4G', '5G', 'Vo5G', 'NFC', 'memory_card_support', 'IR_Blaster']
    
    data_list = []

    for feature in feature_cols:
        ct = pd.crosstab(df[feature], df[target_col], normalize='index')

        yes_score = 0
        if 'Yes' in ct.index and 'expensive' in ct.columns:
            yes_score = ct.loc['Yes', 'expensive']
            
        no_score = 0
        if 'No' in ct.index and 'non-expensive' in ct.columns:
            no_score = ct.loc['No', 'non-expensive']

        data_list.append({
            'Feature': feature,
            'Yes -> Expensive': yes_score,
            'No -> Non-Expensive': no_score
        })

    plot_df = pd.DataFrame(data_list)

    plot_df_melted = plot_df.melt(id_vars='Feature', var_name='Metric', value_name='Percentage')

    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    ax = sns.barplot(
        data=plot_df_melted, 
        x='Feature', 
        y='Percentage', 
        hue='Metric', 
        palette=['#1f77b4', '#ff7f0e']
    )

    plt.title('Impact of Binary Features on Price', fontsize=16)
    plt.ylabel('Percentage', fontsize=12)
    plt.xlabel('Feature', fontsize=12)
    plt.ylim(0, 1.15)
    plt.xticks(rotation=45)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)

    plt.tight_layout()
    plt.show()

def unique(df):
    # Set pandas to display the full list without truncating (optional but helpful)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)

    for col in df.columns:
        print(f"--- Column: {col} ---")
        # This prints the full array of unique values
        print(df[col].unique()) 
        print("\n") # Adds a blank line for readability


sns.set_theme(style="whitegrid")

def analyze_unknown_percentage(df):

    # Calculate percentage
    unknown_counts = df.eq("Unknown").sum()
    unknown_percentages = (unknown_counts / len(df)) * 100
    
    # Filter out columns with 0% unknowns to keep the chart clean
    unknown_percentages = unknown_percentages[unknown_percentages > 0].sort_values(ascending=False)
    
    if unknown_percentages.empty:
        print("No 'Unknown' values found in the DataFrame.")
        return

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=unknown_percentages.index, y=unknown_percentages.values, palette="Reds_r")
    
    plt.title("Percentage of 'Unknown' Values by Column", fontsize=15)
    plt.ylabel("Percentage (%)")
    plt.xlabel("Columns")
    plt.xticks(rotation=45, ha='right') # Rotate labels if there are many columns
    plt.tight_layout()
    plt.show()

def analyze_categorical_variability(df):

    categorical_cols = df.select_dtypes(include=['object']).columns
    variability_data = {}

    for col in categorical_cols:
        if not df[col].empty:
            top_freq = df[col].value_counts(normalize=True).iloc[0]
            variability_data[col] = 1 - top_freq
    
    # Create a Series for plotting
    variability_series = pd.Series(variability_data).sort_values(ascending=False)

    if variability_series.empty:
        print("No categorical columns found.")
        return

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=variability_series.values, y=variability_series.index, palette="viridis")
    
    plt.title("Categorical Column Variability", fontsize=15)
    plt.xlabel("Variability Score (0 to 1)")
    plt.ylabel("Categorical Columns")
    plt.xlim(0, 1) # Fix x-axis from 0 to 1 since it's a probability
    plt.tight_layout()
    plt.show()



def analyze_price_correlation(df, target_col='price'):

    # Check if target exists
    if target_col not in df.columns:
        print(f"Column '{target_col}' not found in DataFrame.")
        return

    # Calculate correlation
    correlation = df.corrwith(df[target_col], numeric_only=True)
    
    # Drop the target column itself (correlation of 1.0 is useless info)
    if target_col in correlation:
        correlation = correlation.drop(target_col)
        
    # Sort values for a clean wave-like chart
    correlation = correlation.sort_values()

    if correlation.empty:
        print("No numeric columns found to correlate.")
        return

    # Visualization
    plt.figure(figsize=(10, 8))
    
    # Create a color map: Blue for negative corr, Red for positive corr
    colors = ['#ef5350' if x > 0 else '#42a5f5' for x in correlation.values]
    
    sns.barplot(x=correlation.values, y=correlation.index, palette=colors)
    
    plt.title(f"Feature Correlation with '{target_col}'", fontsize=15)
    plt.xlabel("Correlation Coefficient")
    plt.axvline(0, color='black', linewidth=1) # Add a line at 0 for reference
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df=pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train.csv")
    df, _, _, _, _ = preprocess(df, train=True)
    df.to_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\preproccessed_train.csv", index=False)
