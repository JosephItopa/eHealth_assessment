import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("climate_data.csv")  # Replace with the actual filename

# Display basic info
print(df.info())
print(df.describe())

# Convert date columns if applicable
df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

# Handle missing values
df.fillna(method='ffill', inplace=True)  # Forward fill

# Remove unknown values
df.replace("UNKNOWN", np.nan, inplace=True)
df.dropna(inplace=True)

# Introduce Wet and Dry Season
# Assuming wet season is from May to October and dry season is from November to April
df['SEASON'] = df['MONTH'].apply(lambda x: 'Wet' if 5 <= x <= 10 else 'Dry')

# Summary statistics
print(df.describe())

# Check for outliers using boxplots
features = ['MERRA-2 Temperature at 2 Meters (C)', 'MERRA-2 Relative Humidity at 2 Meters (%)',
            'MERRA-2 Precipitation Corrected (mm/day)', 'MERRA-2 Wind Speed at 10 Meters (m/s)']

for feature in features:
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df[feature])
    plt.title(f"Boxplot for {feature}")
    plt.show()

# Replace outliers with median
for feature in features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median_value = df[feature].median()
    df[feature] = np.where((df[feature] < lower_bound) | (df[feature] > upper_bound), median_value, df[feature])

# Standardize selected features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Temperature trends
df.groupby("YEAR")['MERRA-2 Temperature at 2 Meters (C)'].mean().plot(
    kind='line', figsize=(10,5), title='Average Temperature Over Years'
)
plt.show()

# Seasonal Analysis
sns.boxplot(x=df['MONTH'], y=df['MERRA-2 Temperature at 2 Meters (C)'])
plt.title("Temperature Distribution by Month")
plt.show()

# Precipitation trends
df.groupby("YEAR")['MERRA-2 Precipitation Corrected (mm/day)'].sum().plot(
    kind='bar', figsize=(12,5), title='Annual Precipitation Over Years'
)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Humidity vs Temperature
sns.scatterplot(x=df['MERRA-2 Temperature at 2 Meters (C)'], y=df['MERRA-2 Relative Humidity at 2 Meters (%)'])
plt.title("Temperature vs Humidity")
plt.show()

# Wet vs Dry Season Analysis
plt.figure(figsize=(8,5))
sns.boxplot(x=df['SEASON'], y=df['MERRA-2 Temperature at 2 Meters (C)'])
plt.title("Temperature Variation in Wet and Dry Seasons")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['SEASON'], y=df['MERRA-2 Precipitation Corrected (mm/day)'])
plt.title("Precipitation in Wet and Dry Seasons")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df['SEASON'], y=df['MERRA-2 Wind Speed at 10 Meters (m/s)'])
plt.title("Wind Speed Comparison Between Seasons")
plt.show()

print("EDA Completed Successfully!")
