from src import Model
from src import Survey
import numpy as np
import pandas as pd
import seaborn as sns
from src import HermesData, compute_leverage
import matplotlib.pyplot as plt
#dummy_model=Model(version="0.0.1",x=np.linspace(0,10,10),y_obs=np.random.normal(5,2,10),
#                  y_err_low=np.random.uniform(0.1,0.5,10),y_err_high=np.random.uniform(0.1,0.5,10))
#dummy_model.build_model(n_samples=500,random_seed=42,centered=True)

# 

filepath = 'dataset/hermes_synthetic_data.csv'

data= HermesData.from_csv(filepath)
leverage_mass=compute_leverage(data.logM)
masses = data.logM.values
plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
plt.hist(data.logM, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of logM')

plt.subplot(1,2,2)
plt.boxplot(data.logM, vert=False)
plt.title("Boxplot")

plt.tight_layout()
plt.show()

print("Summary statistics:")
print("-------------------")
print(f"Mean:     {np.mean(masses):.4f}")
print(f"Median:   {np.median(masses):.4f}")
print(f"Min:      {np.min(masses):.4f}")
print(f"Max:      {np.max(masses):.4f}")
print(f"Std:      {np.std(masses):.4f}")
print("Quantiles:", np.quantile(masses, [0, 0.25, 0.5, 0.75, 1]))