import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the dataset again to ensure fresh start
df = pd.read_csv('dataset (1).csv', sep=';')

# Mapping for ATECO Sections (Standard Italian ATECO first letter)
ateco_map = {
    'A': 'Agriculture', 'B': 'Mining', 'C': 'Manufacturing', 'D': 'Energy',
    'E': 'Water/Waste', 'F': 'Construction', 'G': 'Wholesale/Retail',
    'H': 'Transport/Logistics', 'I': 'Accommodation/Food', 'J': 'Information/Comm',
    'K': 'Finance/Insurance', 'L': 'Real Estate', 'M': 'Professional/Tech',
    'N': 'Administrative/Support', 'O': 'Public Admin', 'P': 'Education',
    'Q': 'Health/Social', 'R': 'Arts/Entertainment', 'S': 'Other Services'
}

df['Sector_Code'] = df['ATECO'].str[0]
df['Sector_Name'] = df['Sector_Code'].map(ateco_map).fillna('Other/Unknown')

# Time columns M-35 (oldest) to M-0 (newest)
time_cols = [f'M-{i}' for i in range(35, -1, -1)]

# --- 1. TOTAL REVENUE FORECASTING ---
# Aggregate total revenue per month
monthly_total = df[time_cols].sum()
monthly_total.index = range(36) # 0 to 35

# Prepare data for Linear Regression (Trend + Seasonality)
X = pd.DataFrame({'Month': range(36)})
X['Month_of_Year'] = X['Month'] % 12
# Dummy variables for seasonality
X = pd.concat([X, pd.get_dummies(X['Month_of_Year'], prefix='Month')], axis=1)

y = monthly_total.values

model = LinearRegression()
model.fit(X.drop('Month_of_Year', axis=1), y)

# Predict next 6 months (36 to 41)
future_months = pd.DataFrame({'Month': range(36, 42)})
future_months['Month_of_Year'] = future_months['Month'] % 12
future_months = pd.concat([future_months, pd.get_dummies(future_months['Month_of_Year'], prefix='Month')], axis=1)

# Ensure all month columns exist (in case some were missing in the training set, though unlikely for 36 months)
for col in X.columns:
    if col not in future_months.columns:
        future_months[col] = 0

forecast = model.predict(future_months[X.drop('Month_of_Year', axis=1).columns])

# --- 2. CUSTOMER SEGMENTATION (CLUSTERING) ---
# Features for clustering: Total Volume, Average Order, Volatility (Std Dev), Fatturato (Company Size)
customer_features = df.copy()
customer_features['Total_Volume'] = df[time_cols].sum(axis=1)
customer_features['Avg_Order'] = df[time_cols].mean(axis=1)
customer_features['Order_Volatility'] = df[time_cols].std(axis=1)

# Select and scale features
features = ['Total_Volume', 'Avg_Order', 'Order_Volatility', 'Fatturato']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features[features].fillna(0))

# K-Means
kmeans = KMeans(n_components=4, random_state=42)
customer_features['Cluster'] = kmeans.fit_predict(scaled_features)

# --- 3. VISUALIZATIONS ---
# Forecast Plot
plt.figure(figsize=(12, 6))
plt.plot(range(36), y, label='Historical Revenue', marker='o')
plt.plot(range(35, 42), np.concatenate([[y[-1]], forecast]), label='6-Month Forecast', linestyle='--', color='red', marker='s')
plt.title('Monthly Total Revenue Forecast (M-35 to M+6)')
plt.xlabel('Month Index')
plt.ylabel('Total Order Volume')
plt.legend()
plt.grid(True)
plt.savefig('revenue_forecast.png')

# Clustering Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_features, x='Fatturato', y='Total_Volume', hue='Cluster', palette='viridis', alpha=0.6)
plt.xscale('log')
plt.yscale('log')
plt.title('Customer Segmentation: Revenue vs Total Order Volume')
plt.savefig('segmentation_clusters.png')

# Save segments to CSV
customer_features[['ATECO', 'Sector_Name', 'Provincia', 'Fatturato', 'Total_Volume', 'Avg_Order', 'Cluster']].to_csv('detailed_segmentation.csv', index=False)

print("Forecast for next 6 months:")
print(forecast)
print("\nCluster Distribution:")
print(customer_features['Cluster'].value_counts())