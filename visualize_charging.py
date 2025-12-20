import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train_imputed.csv")

# Filter out extreme outliers (e.g., 0W or >150W) just for cleaner plotting if needed
# but for now, let's see raw data.

plt.figure(figsize=(12, 6))

# We use a Scatter Plot with 'Jitter' (strip plot) to see density
# hue='price' colors the dots by price (assuming price is categorical/binned)
sns.stripplot(x='price', y='fast_charging_power', data=df, jitter=True, alpha=0.6, palette='viridis')

plt.title('The Charging Trap: High Watts on Cheaper Phones?')
plt.xlabel('Price Category (0=Cheap, 1=Expensive, etc)')
plt.ylabel('Charging Power (Watts)')
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add a line at 45W (Standard "Flagship" speed for Samsung/Apple)
plt.axhline(y=45, color='r', linestyle='--', label='45W (Flagship Std)')
plt.legend()

plt.show()