import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\RexoL\source\repos\Smart-Phone-Prices-Prediction\train.csv")

plt.figure(figsize=(12, 6))


sns.stripplot(x='price', y='fast_charging_power', data=df, jitter=True, alpha=0.6, palette='viridis')

plt.title('The Charging Trap: High Watts on Cheaper Phones?')
plt.xlabel('Price Category (0=Cheap, 1=Expensive, etc)')
plt.ylabel('Charging Power (Watts)')
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.axhline(y=45, color='r', linestyle='--', label='45W')
plt.legend()

plt.show()