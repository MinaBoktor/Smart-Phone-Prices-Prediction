import pandas as pd

column_name = "brand"
df = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train.csv")


column_data_list = df[column_name].tolist()

# --- 3. Print the result ---
print(f"Data from the '{column_name}' column as a list:")
print(column_data_list)